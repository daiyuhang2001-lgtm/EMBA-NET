"""Focused regression checks; run python -m unittest discover -s tests -v."""
import tempfile,unittest
from pathlib import Path
import torch
from emba_net import EMBANet
from emba_net.loss import DetectionLoss
from emba_net.inference import non_max_suppression

CONFIG=Path(__file__).resolve().parents[1]/"configs/EMBA-Net-baseline.yaml"

class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.model=EMBANet(CONFIG,nc=1,scale="n")

    def test_graph_and_shapes(self):
        m=self.model.eval()
        self.assertEqual(sum(p.numel() for p in m.parameters()),2314132)
        self.assertEqual(m.stride.tolist(),[4.,8.,16.])
        with torch.inference_mode():pred,raw=m(torch.zeros(1,3,128,256))
        self.assertEqual(tuple(pred.shape),(1,5,2688))
        self.assertEqual([tuple(x.shape[-2:]) for x in raw],[(32,64),(16,32),(8,16)])
        self.assertTrue(torch.isfinite(pred).all())

    def test_positive_and_background_backward(self):
        m=self.model.train()
        criterion=DetectionLoss(m)
        for empty in (False,True):
            batch={"batch_idx":torch.empty(0) if empty else torch.tensor([0.,1.]),
                   "cls":torch.empty(0,1) if empty else torch.zeros(2,1),
                   "bboxes":torch.empty(0,4) if empty else torch.tensor([[.5,.5,.3,.3],[.6,.4,.2,.2]])}
            loss,parts=criterion(m(torch.rand(2,3,64,64)),batch)
            self.assertTrue(torch.isfinite(loss))
            m.zero_grad();loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    def test_strict_tensor_checkpoint(self):
        m=self.model.eval()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"weights.pt"
            m.save_weights(path)
            n=EMBANet(CONFIG,nc=1,scale="n").load_weights(path).eval()
            with torch.inference_mode():
                x=torch.rand(1,3,64,64)
                self.assertTrue(torch.equal(m(x)[0],n(x)[0]))
            with self.assertRaises(RuntimeError):EMBANet(CONFIG,nc=2,scale="n").load_weights(path)

    def test_class_aware_suppression(self):
        # Two overlapping class-0 boxes and an overlapping class-1 box.
        p=torch.tensor([[[50.,51.,50.],[50.,51.,50.],[20.,20.,20.],[20.,20.,20.],
                         [.9,.8,.1],[.1,.1,.85]]])
        d=non_max_suppression(p,conf=.5,iou=.5)[0]
        self.assertEqual(len(d),2)
        self.assertEqual(set(d[:,5].tolist()),{0.,1.})

    def test_legacy_weights_and_paper_names(self):
        m=self.model.eval()
        modules=[row["module"] for row in m.summary()["layers"]]
        self.assertEqual(modules.count("BGSA"),5)
        self.assertEqual(modules.count("GLM_P_B"),5)
        self.assertEqual(modules[-1],"LSDheadreplacement")
        self.assertEqual(next(row for row in m.summary()["layers"] if row["index"]==11)["paper_name"],"GLM-P B")
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"legacy.pt"
            legacy={k.replace(".context.",".EMA."):v for k,v in m.state_dict().items()}
            torch.save({"state_dict":legacy},path)
            n=EMBANet(CONFIG,nc=1,scale="n").load_weights(path).eval()
            with torch.inference_mode():
                x=torch.rand(1,3,64,64)
                self.assertTrue(torch.equal(m(x)[0],n(x)[0]))

    def test_backbone_features_and_flat_legacy_checkpoint(self):
        m=self.model.eval()
        self.assertEqual(m.summary()["layers"][0]["module"],"PVFeaturePerceptionNet")
        with torch.inference_mode():features=m.model.backbone(torch.zeros(1,3,128,128))
        self.assertEqual(list(features),[2,4,6,9])
        self.assertEqual([tuple(t.shape[-2:]) for t in features.values()],[(32,32),(16,16),(8,8),(4,4)])
        legacy={}
        for key,value in m.state_dict().items():
            if key.startswith("model.backbone.layers."):
                key="model."+key[len("model.backbone.layers."):]
            elif key.startswith("model.head."):
                index,suffix=key[len("model.head."):].split(".",1)
                key=f"model.{int(index)+10}.{suffix}"
            legacy[key.replace(".context.",".EMA.")]=value
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"flat_legacy.pt";torch.save({"state_dict":legacy},path)
            n=EMBANet(CONFIG,nc=1,scale="n").load_weights(path).eval()
            with torch.inference_mode():
                x=torch.rand(1,3,128,128)
                self.assertTrue(torch.equal(m(x)[0],n(x)[0]))

if __name__=="__main__":unittest.main()
