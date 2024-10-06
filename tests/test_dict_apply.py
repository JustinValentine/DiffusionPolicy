from diffusion_policy.common.pytorch_util import dict_apply
import torch

def test_dict_apply():
    no_dict = torch.tensor(1)
    assert dict_apply(no_dict, lambda x: x) == no_dict

    single_dict = {'a': torch.tensor(1)}
    assert dict_apply(single_dict, lambda x: x+1) == {'a': torch.tensor(2)}

    nested_dict = {'a': {'b': torch.tensor(1)}}
    assert dict_apply(nested_dict, lambda x: x+1) == {'a': {'b': torch.tensor(2)}}
    
    multiple_nested_dict = {'a': {'b': torch.tensor(1)}, 'c': {'d': torch.tensor(2)}}
    assert dict_apply(multiple_nested_dict, lambda x: x+1) == {'a': {'b': torch.tensor(2)}, 'c': {'d': torch.tensor(3)}}
