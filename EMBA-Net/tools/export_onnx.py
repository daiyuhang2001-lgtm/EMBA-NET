"""Optional fixed-shape ONNX export; requires requirements-export.txt."""
import argparse,sys
from types import MethodType
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from emba_net import EMBANet
from emba_net.glm_pb import GLMContext


def _fixed_shape_glm_context(self, x):
    """Equivalent directional averages with explicit shapes for legacy ONNX export.

    AdaptiveAvgPool2d((None,1)) can lose channel/shape information in the
    legacy exporter. This export-only path uses the same arithmetic and weights
    with a fixed input shape. Normal training/inference forward is untouched.
    """
    b,c,h,w=map(int,x.shape)
    gx=x.reshape(b*self.groups,c//self.groups,h,w)
    xh=gx.mean(3,keepdim=True)
    xw=gx.mean(2,keepdim=True).permute(0,1,3,2)
    hw=self.conv1x1(torch.cat((xh,xw),dim=2))
    xh,xw=torch.split(hw,[h,w],dim=2)
    x1=self.gn(gx*xh.sigmoid()*xw.permute(0,1,3,2).sigmoid())
    x2=self.conv3x3(gx)
    x11=self.softmax(x1.mean((2,3),keepdim=True).reshape(b*self.groups,c//self.groups,1).permute(0,2,1))
    x12=x2.reshape(b*self.groups,c//self.groups,h*w)
    x21=self.softmax(x2.mean((2,3),keepdim=True).reshape(b*self.groups,c//self.groups,1).permute(0,2,1))
    x22=x1.reshape(b*self.groups,c//self.groups,h*w)
    weights=(torch.matmul(x11,x12)+torch.matmul(x21,x22)).reshape(b*self.groups,1,h,w)
    return (gx*weights.sigmoid()).reshape(b,c,h,w)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default=str(Path(__file__).resolve().parents[1]/"configs/EMBA-Net.yaml"))
    p.add_argument("--weights",required=True)
    p.add_argument("--output",default="emba_net.onnx")
    p.add_argument("--nc",type=int,default=1)
    p.add_argument("--scale",default="n",choices=list("nslmx"))
    p.add_argument("--imgsz",type=int,default=640)
    a=p.parse_args()
    model=EMBANet(a.config,nc=a.nc,scale=a.scale).load_weights(a.weights).eval().switch_to_deploy()
    model.model[-1].export=True
    model.model[-1].format="onnx"
    for module in model.modules():
        if isinstance(module,GLMContext):module.forward=MethodType(_fixed_shape_glm_context,module)
    torch.onnx.export(model,torch.zeros(1,3,a.imgsz,a.imgsz),a.output,opset_version=17,
                      input_names=["images"],output_names=["predictions"],do_constant_folding=True)
    import onnx
    onnx.checker.check_model(onnx.load(a.output))
    print(f"Validated fixed-shape ONNX: {a.output}; NMS is external.")

if __name__=="__main__":main()
