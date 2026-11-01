import copy,tempfile,unittest
from pathlib import Path
import torch
from emba_net import EMBANet
from emba_net.repconv import RepConv
from emba_net.loss import DetectionLoss
from emba_net.complexity import profile_model

CONFIG=Path(__file__).resolve().parents[1]/'configs/EMBA-Net.yaml'


class PaperAlignmentChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_repconv_fusion_with_nontrivial_bn(self):
        torch.manual_seed(41)
        for c2,stride in [(8,1),(12,2)]:
            layer=RepConv(8,c2,stride=stride,branches=2).train()
            for _ in range(3):layer(torch.randn(2,8,20,24))
            layer.eval();x=torch.randn(2,8,20,24)
            with torch.no_grad():expected=layer(x)
            layer.switch_to_deploy()
            torch.testing.assert_close(layer(x),expected,rtol=2e-5,atol=2e-6)
            self.assertFalse(hasattr(layer,'dense'))

    def test_budget_and_actual_operators(self):
        model=EMBANet(CONFIG).eval()
        report=profile_model(model)
        self.assertTrue(8.72<=report['deployment_gflops']<8.74)
        self.assertLess(abs(report['training_parameters']-2314132)/2314132,0.01)
        self.assertEqual(sum(isinstance(m,RepConv) for m in model.modules()),10)
        depthwise=[m for m in model.model[-1].modules() if isinstance(m,torch.nn.Conv2d) and m.groups==m.in_channels and m.in_channels>1]
        self.assertEqual(len(depthwise),5)

    def test_all_trainable_parameters_receive_gradients(self):
        torch.manual_seed(23);model=EMBANet(CONFIG).train()
        criterion=DetectionLoss(model)
        batch={'batch_idx':torch.tensor([0.,1.]),'cls':torch.zeros(2,1),
               'bboxes':torch.tensor([[.4,.5,.2,.2],[.6,.5,.3,.25]])}
        loss,_=criterion(model(torch.rand(2,3,96,128)),batch)
        loss.backward()
        missing=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
        self.assertEqual(missing,[])
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_full_model_fusion_and_both_checkpoint_modes(self):
        torch.manual_seed(7);model=EMBANet(CONFIG).eval()
        x=torch.randn(1,3,128,256)
        with torch.inference_mode():expected=model(x)[0]
        deploy=copy.deepcopy(model).switch_to_deploy()
        with torch.inference_mode():actual=deploy(x)[0]
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-4)
        self.assertEqual(actual.shape,(1,5,2688))
        with tempfile.TemporaryDirectory() as td:
            for name,source in [('training',model),('deployment',deploy)]:
                path=Path(td)/(name+'.pt');source.save_weights(path)
                loaded=EMBANet(CONFIG).load_weights(path).eval()
                with torch.inference_mode():
                    restored=loaded(x)[0]
                    reference=source(x)[0]
                torch.testing.assert_close(restored,reference,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
