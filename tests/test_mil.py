import torch
from model import MIL

class TestMIL:
    def test_init(self):
        mil = MIL()
        assert mil is not None

    def test_term2(self):
        mil = MIL()
        with torch.no_grad():
            anomalous = torch.tensor([
                1.0, 1.0
            ])
            assert mil._term2(anomalous).item() == 2.0

            anomalous = torch.tensor([
                2.0, 3.0, 5.0
            ])
            assert mil._term2(anomalous).item() == 10.0

            anomalous = torch.tensor([
                5.0, 3.0, 2.0
            ])
            assert mil._term2(anomalous).item() == 10.0


    def test_term1(self):
        mil = MIL()
        with torch.no_grad():
            anomalous = torch.tensor([
                1.0, 1.0
            ])
            assert mil._term1(anomalous).item() == 0.0

            # 1.0, 4.0 = 5.0
            anomalous = torch.tensor([
                2.0, 3.0, 5.0
            ])
            assert mil._term1(anomalous).item() == 5.0

            # 4.0, 1.0 = 5.0
            anomalous = torch.tensor([
                5.0, 3.0, 2.0
            ])
            assert mil._term1(anomalous).item() == 5.0


    def test_term0(self):
        mil = MIL()
        with torch.no_grad():
            anomalous = torch.tensor([
                1.0, 1.0
            ])
            normal = torch.tensor([
                1.0, 1.0
            ])
            assert mil._term0(anomalous, normal).item() == 1.0

            anomalous = torch.tensor([
                1.0, 10.0
            ])
            normal = torch.tensor([
                1.0, 1.0
            ])
            assert mil._term0(anomalous, normal).item() == 0.0

            anomalous = torch.tensor([
                10.0, 1.0
            ])
            normal = torch.tensor([
                1.0, 1.0
            ])
            assert mil._term0(anomalous, normal).item() == 0.0

            anomalous = torch.tensor([
                1.0, 1.0
            ])
            normal = torch.tensor([
                1.0, 10.0
            ])
            assert mil._term0(anomalous, normal).item() == 10.0