"""Frozen ProtT5 embeddings must not disconnect trainable LoRA parameters."""
import torch
import pytest
from transformers import T5Config,T5EncoderModel
from ghxtox.lora_mamba_1d import build_model

@pytest.mark.parametrize('legacy_api',[False,True])
def test_checkpointed_encoder_updates_lora_with_frozen_embeddings(monkeypatch,legacy_api):
    torch.manual_seed(42)
    plm=T5EncoderModel(T5Config(vocab_size=33,d_model=32,d_kv=8,d_ff=64,num_layers=2,num_heads=4,dropout_rate=0.0))
    if legacy_api:
        original=plm.gradient_checkpointing_enable
        monkeypatch.setattr(plm,'gradient_checkpointing_enable',lambda:original(gradient_checkpointing_kwargs={'use_reentrant':True}))
    config=dict(token_dim=32,d_model=16,d_state=4,d_conv=2,expand=1,n_layers=1,lora_rank=2,lora_alpha=4,lora_dropout=0,dropout=0,grad_checkpoint=True)
    model=build_model(plm,config,torch.device('cpu'))
    assert not plm.get_input_embeddings().weight.requires_grad
    model.train()
    ids=torch.tensor([[3,4,5,1],[6,7,8,1]])
    logits=model(ids,torch.ones_like(ids),torch.tensor([3,3]))
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([0.,1.]))
    loss.backward()
    lora_b=[p for n,p in model.named_parameters() if '.lora_B' in n]
    assert lora_b and all(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in lora_b)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert torch.count_nonzero(model.classifier.weight.grad)>0
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
    optimizer.step()
    assert all(torch.count_nonzero(p)>0 for p in lora_b)
    # Checkpointing changes backward storage, not the evaluation function.
    model.eval()
    with torch.no_grad(): before=model(ids,torch.ones_like(ids),torch.tensor([3,3]))
    plm.gradient_checkpointing_disable()
    with torch.no_grad(): after=model(ids,torch.ones_like(ids),torch.tensor([3,3]))
    torch.testing.assert_close(before,after,rtol=0,atol=0)


def test_guard_detects_disconnected_adapters_and_optimizer_rates():
    from ghxtox.lora_mamba_1d import check_lora_gradients, make_training_optimizer
    plm=T5EncoderModel(T5Config(vocab_size=33,d_model=32,d_kv=8,d_ff=64,num_layers=1,num_heads=4))
    model=build_model(plm,dict(token_dim=32,d_model=16,d_state=4,d_conv=2,expand=1,n_layers=1,lora_rank=2),torch.device('cpu'))
    with pytest.raises(RuntimeError,match='gradient disconnected'):
        check_lora_gradients(model)
    opt=make_training_optimizer(model,dict(learning_rate=3e-4,lora_learning_rate=1e-5))
    assert {g['name']:g['lr'] for g in opt.param_groups}=={'head':3e-4,'lora':1e-5}
    expected={id(p) for p in model.parameters() if p.requires_grad}
    actual=[id(p) for g in opt.param_groups for p in g['params']]
    assert len(actual)==len(set(actual)) and set(actual)==expected


def test_rank_zero_removes_adapters_and_preserves_frozen_encoder():
    from ghxtox.lora_mamba_1d import wrap_encoder_lora, LoRALinear
    import copy
    plm=T5EncoderModel(T5Config(vocab_size=33,d_model=32,d_kv=8,d_ff=64,num_layers=1,num_heads=4,dropout_rate=0.0))
    other=copy.deepcopy(plm)
    assert wrap_encoder_lora(plm,rank=0)==0
    assert all(not p.requires_grad for p in plm.parameters())
    assert not any(isinstance(m,LoRALinear) for m in plm.modules())
    plm.eval();other.eval();ids=torch.tensor([[3,4,5,1]])
    with torch.no_grad():torch.testing.assert_close(plm(ids).last_hidden_state,other(ids).last_hidden_state,rtol=0,atol=0)


@pytest.mark.parametrize('adapter_lr',[0.0,1e-3])
def test_training_entry_rejects_adapter_without_actual_updates(adapter_lr):
    from ghxtox.lora_mamba_1d import train_epoch,make_training_optimizer
    plm=T5EncoderModel(T5Config(vocab_size=33,d_model=32,d_kv=8,d_ff=64,num_layers=1,num_heads=4,dropout_rate=0.0))
    config=dict(token_dim=32,d_model=16,d_state=4,d_conv=2,expand=1,n_layers=1,dropout=0,lora_rank=2,lora_dropout=0,grad_checkpoint=True,lora_learning_rate=adapter_lr)
    model=build_model(plm,config,torch.device('cpu'))
    optimizer=make_training_optimizer(model,config)
    batch={'input_ids':torch.tensor([[3,4,5,1],[6,7,8,1]]),'attention_mask':torch.ones(2,4,dtype=torch.long),'length':torch.tensor([3,3]),'label':torch.tensor([0.,1.])}
    args=(model,[batch],optimizer,torch.nn.BCEWithLogitsLoss(),torch.device('cpu'),False,None)
    if adapter_lr==0:
        with pytest.raises(RuntimeError,match='did not update'):train_epoch(*args)
    else:train_epoch(*args)


def test_legacy_zero_adapters_are_removed_without_changing_output():
    from ghxtox.lora_mamba_1d import remove_inactive_lora,LoRALinear
    plm=T5EncoderModel(T5Config(vocab_size=33,d_model=32,d_kv=8,d_ff=64,num_layers=1,num_heads=4,dropout_rate=0.0))
    model=build_model(plm,dict(token_dim=32,d_model=16,d_state=4,d_conv=2,expand=1,n_layers=1,lora_rank=2,dropout=0),torch.device('cpu')).eval()
    ids=torch.tensor([[3,4,5,1]]);mask=torch.ones_like(ids);length=torch.tensor([3])
    with torch.no_grad():before=model(ids,mask,length)
    assert remove_inactive_lora(model.plm.encoder)>0
    assert not any(isinstance(x,LoRALinear) for x in model.modules())
    with torch.no_grad():after=model(ids,mask,length)
    torch.testing.assert_close(before,after,rtol=0,atol=0)


def test_active_adapter_rejected_without_partial_removal():
    from ghxtox.lora_mamba_1d import remove_inactive_lora,LoRALinear
    plm=T5EncoderModel(T5Config(vocab_size=33,d_model=32,d_kv=8,d_ff=64,num_layers=1,num_heads=4))
    build_model(plm,dict(token_dim=32,d_model=16,d_state=4,d_conv=2,expand=1,n_layers=1,lora_rank=2),torch.device('cpu'))
    modules=[m for m in plm.encoder.modules() if isinstance(m,LoRALinear)]
    with torch.no_grad():modules[-1].lora_B[0,0]=1
    with pytest.raises(ValueError,match='active LoRA'):remove_inactive_lora(plm.encoder)
    assert len([m for m in plm.encoder.modules() if isinstance(m,LoRALinear)])==len(modules)


def test_no_lora_checkpoint_contract_accepts_valid_and_rejects_adapter():
    from ghxtox.lora_mamba_1d import validate_no_lora_checkpoint_payload
    plm=T5EncoderModel(T5Config(vocab_size=33,d_model=32,d_kv=8,d_ff=64,num_layers=1,num_heads=4))
    cfg=dict(token_dim=32,d_model=16,d_state=4,d_conv=2,expand=1,n_layers=1,lora_rank=0)
    model=build_model(plm,cfg,torch.device('cpu'))
    state={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    payload={'config':cfg,'model_state':state}
    validate_no_lora_checkpoint_payload(payload,cfg,model=model)
    payload['model_state']['plm.encoder.fake.lora_B']=torch.zeros(1)
    with pytest.raises(ValueError,match='invalid no-LoRA'):validate_no_lora_checkpoint_payload(payload,cfg,model=model)
