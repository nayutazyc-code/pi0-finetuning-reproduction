"""Extra proxy timing must not change canonical pixels, state, RTC or actions."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from openpi_client import msgpack_numpy
import io,hashlib
from types import SimpleNamespace

spec=importlib.util.spec_from_file_location('instrumented_proxy',Path(__file__).resolve().parents[1]/'server/image_transport_proxy.py')
proxy=importlib.util.module_from_spec(spec);spec.loader.exec_module(proxy)


class ProxyTimingTests(unittest.IsolatedAsyncioTestCase):
    async def test_echo_is_bounded_and_does_not_open_model_connection(self):
        payload=b'probe'*11000
        class Front:
            request=SimpleNamespace(path='/transport-probe')
            def __init__(self):self.sent=[];self.index=0;self.closed=None
            def __aiter__(self):return self
            async def __anext__(self):
                self.index+=1
                if self.index==1:return payload
                if self.index==2:return b'too small'
                raise StopAsyncIteration
            async def send(self,data):self.sent.append(data)
            async def close(self,**kwargs):self.closed=kwargs
        front=Front()
        with patch.object(proxy,'connect',side_effect=AssertionError('Echo must not reach model')):
            await proxy.handler(front,8000)
        self.assertEqual(msgpack_numpy.unpackb(front.sent[0])['schema'],'pi0-transport-echo-v1')
        self.assertEqual(front.sent[1],payload[:8192]);self.assertEqual(front.closed['code'],1008)
    async def test_png_protocol_values_identical_with_finite_nested_timing(self):
        packer=msgpack_numpy.Packer();image=np.zeros((3,480,640),np.uint8);image[:,100:200,200:300]=123
        buffer=io.BytesIO();Image.fromarray(image.transpose(1,2,0)).save(buffer,format='PNG')
        prefix=np.arange(1400,dtype=np.float32).reshape(100,14)
        observation={'state':np.arange(14,dtype=np.float32),'prompt':'banana','rtc':{'prefix':prefix,'blend_steps':20},
            'images_png':{'cam_high':buffer.getvalue()},'image_sha256':{'cam_high':hashlib.sha256(image.tobytes()).hexdigest()}}
        action=np.arange(1400,dtype=np.float32).reshape(100,14)
        class Upstream:
            def __init__(self):self.sent=[];self.receives=0
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            async def send(self,data):self.sent.append(msgpack_numpy.unpackb(data))
            async def recv(self):
                self.receives+=1
                return packer.pack({'model_family':'pi0'} if self.receives==1 else {'actions':action,'rtc':{'schema':'pi0-rtc-v1'}})
        class Front:
            def __init__(self):self.sent=[]
            def __aiter__(self):return self
            async def __anext__(self):
                if len(self.sent)>1:raise StopAsyncIteration
                return packer.pack(observation)
            async def send(self,data):self.sent.append(msgpack_numpy.unpackb(data))
        upstream=Upstream();front=Front()
        with patch.object(proxy,'connect',return_value=upstream):await proxy.handler(front,8000)
        self.assertEqual(len(upstream.sent),1);canonical=upstream.sent[0]
        np.testing.assert_array_equal(canonical['images']['cam_high'],image)
        np.testing.assert_array_equal(canonical['state'],observation['state'])
        np.testing.assert_array_equal(canonical['rtc']['prefix'],prefix)
        self.assertEqual(canonical['rtc']['blend_steps'],20)
        np.testing.assert_array_equal(front.sent[1]['actions'],action)
        timing=front.sent[1]['transport_timing']
        self.assertTrue(timing['pixel_digest_verified'])
        self.assertEqual(timing['timing_schema'],'pi0-proxy-timing-v1')
        self.assertGreaterEqual(timing['proxy_handler_ms_before_reply_pack'],timing['proxy_upstream_round_trip_ms'])
        self.assertTrue(np.isfinite([timing[k] for k in ('decode_ms','request_unpack_ms','proxy_upstream_round_trip_ms','proxy_handler_ms_before_reply_pack')]).all())


if __name__=='__main__':unittest.main()
