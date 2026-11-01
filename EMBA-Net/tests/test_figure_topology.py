"""Behavioral checks for the exact connections discussed in the figure audit."""
import unittest
from pathlib import Path
import torch
from torch import nn
from emba_net.glm_paper import GLM_P_B
from emba_net.gated_projection import GatedProjectionGN
from emba_net.figure_bgsa import BGSA
from emba_net import EMBANet
from emba_net.loss import DetectionLoss

CONFIG=Path(__file__).resolve().parents[1]/'configs/EMBA-Net.yaml'


class FigureTopologyChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_glm_ema_precedes_split_and_all_bottleneck_taps_are_concatenated(self):
        block=GLM_P_B(16,16,n=3,e=0.5,rep_branches=1).eval()
        seen={};handles=[]
        def save(name):
            def hook(module,args,out):seen[name]=out.detach().clone()
            return hook
        handles.append(block.context.register_forward_hook(save('ema')))
        for i,m in enumerate(block.m):handles.append(m.register_forward_hook(save(f'local{i}')))
        def capture(module,args):seen['concat']=args[0].detach().clone()
        handles.append(block.cv2.register_forward_pre_hook(capture))
        with torch.no_grad():block(torch.randn(2,16,12,16))
        for h in handles:h.remove()
        self.assertEqual(seen['ema'].shape[1],16)
        expected=torch.cat([seen['ema'][:,:8],seen['local0'],seen['local1'],seen['local2']],dim=1)
        torch.testing.assert_close(seen['concat'],expected,rtol=0,atol=0)
        self.assertEqual(block.cv2.c1,32)
        self.assertTrue(isinstance(block.cv1.act,nn.ReLU))

    def test_all_three_figure4_multiplications_are_effective(self):
        torch.manual_seed(55)
        block=GatedProjectionGN(12,16).eval();x=torch.randn(2,12,12,16)
        self.assertEqual(block.gate_widths,(8,16,32))
        with torch.no_grad():normal=block(x)
        self.assertGreater(float(normal.abs().max()),0)
        start=0
        for width in block.gate_widths:
            def suppress(module,args,out,lo=start,hi=start+width):
                result=out.clone();result[:,lo:hi]=0;return result
            handle=block.dsconv.register_forward_hook(suppress)
            with torch.no_grad():suppressed=block(x)
            handle.remove()
            torch.testing.assert_close(suppressed,torch.zeros_like(suppressed),rtol=0,atol=0)
            start+=width

    def test_bgsa_updates_both_streams_from_original_aligned_features(self):
        block=BGSA([4,4],4).eval();block.conv=nn.Identity()
        with torch.no_grad():
            block.fc1.weight.zero_();block.fc2.weight.zero_();block.psi.weight.zero_();block.psi.bias.zero_()
            for i in range(2):block.fc1.weight[i,i,0,0]=1;block.fc2.weight[i,i,0,0]=1
            actual=block([torch.full((1,4,2,2),2.),torch.full((1,4,4,4),6.)])
        expected=torch.cat((torch.full((1,2,4,4),7.),torch.full((1,2,4,4),5.)),dim=1)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_lsd_shared_encoder_runs_at_three_native_resolutions(self):
        model=EMBANet(CONFIG).eval();head=model.model[-1];seen=[]
        handle=head.share_conv.register_forward_pre_hook(lambda module,args:seen.append(tuple(args[0].shape[-2:])))
        with torch.inference_mode():prediction,raw=model(torch.rand(1,3,128,256))
        handle.remove()
        self.assertEqual(seen,[(32,64),(16,32),(8,16)])
        self.assertEqual(tuple(prediction.shape),(1,5,2688))
        self.assertEqual(model.stride.tolist(),[4.,8.,16.])

    def test_clipped_optimizer_step_keeps_eval_outputs_finite(self):
        torch.manual_seed(0);model=EMBANet(CONFIG).train()
        optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.937,weight_decay=.0005)
        batch={'batch_idx':torch.tensor([0.,1.]),'cls':torch.zeros(2,1),
               'bboxes':torch.tensor([[.4,.5,.2,.2],[.6,.5,.3,.25]])}
        loss,_=DetectionLoss(model)(model(torch.rand(2,3,128,128)),batch)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
        model.eval()
        with torch.inference_mode():output=model(torch.rand(1,3,128,128))[0]
        self.assertTrue(torch.isfinite(output).all())


if __name__=='__main__':unittest.main()
