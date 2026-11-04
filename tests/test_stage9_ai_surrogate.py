import pytest
torch = pytest.importorskip("torch")
from pinn_heat_transfer_hybrid import HybridPINN

def test_stage9_ai_surrogate_shape_and_finiteness():
    m=HybridPINN(16,2)
    z=torch.linspace(0,1,7,dtype=torch.float64)[:,None]
    J,P,T,x,jt=m(z)
    assert J.shape==(7,3)
    assert P.shape==(7,)
    assert T.shape==(7,)
    assert x.shape==(7,3)
    assert jt.shape==(7,)
    assert torch.isfinite(J).all()
    assert torch.isfinite(P).all()
    assert torch.isfinite(T).all()
    assert torch.allclose(x.sum(dim=1),torch.ones(7,dtype=torch.float64),atol=1e-12)
