import torch
from canconv.models.cannet import CANNet

if __name__ == '__main__':
    device = "cuda:0"
    N = CANNet().to(device)
    from fvcore.nn import FlopCountAnalysis, parameter_count

    lms = torch.randn((1, 8, 64, 64)).to(device)
    pan = torch.randn((1, 1, 64, 64)).to(device)
    flops = FlopCountAnalysis(N, (pan, lms))
    print("FLOPs(fvcore): ", flops.total(), "=", f"{flops.total() / 1e9}G")
    print("Paras(fvcore): ", parameter_count(N)[''], "=", f"{parameter_count(N)[''] / 1e6}M")
    hr = N(lms, pan)
    print(hr.shape)

    from thop import profile

    flopsTP, paramsTP = profile(N, inputs=(pan, lms,))
    print("FLOPs(thop): ", flopsTP, "=", f"{flopsTP / 1e9}G")
    print("Paras(thop): ", paramsTP, "=", f"{paramsTP / 1e6}M")