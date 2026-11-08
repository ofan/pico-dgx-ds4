"""Host ingress regressions; no GPU or model prompts."""
import base64, io, socket, unittest
from unittest.mock import patch
from PIL import Image
import image_io as m
class Ingress(unittest.TestCase):
    def setUp(self):
        b=io.BytesIO(); Image.new('RGB',(4,3)).save(b,format='PNG');self.raw=b.getvalue();self.data=base64.b64encode(self.raw).decode()
    def test_data(self):self.assertEqual(m.fetch({'url':'data:image/png;base64,'+self.data}),self.raw)
    def test_record(self):self.assertEqual(m.fetch({'data':self.data}),self.raw)
    def test_anthropic(self):self.assertEqual(m.fetch({'source':{'type':'base64','media_type':'image/png','data':self.data}}),self.raw)
    def test_bad_base64(self):
        with self.assertRaises(ValueError):m.fetch({'data':'AAA!'})
    def test_mime_mismatch(self):
        with self.assertRaises(ValueError):m.fetch({'url':'data:image/jpeg;base64,'+self.data})
    def test_local(self):
        for u in ('file:///etc/passwd','/etc/passwd','ftp://public.example/img'):
            with self.assertRaises(ValueError):m.fetch({'url':u})
    def test_pixel_cap(self):
        with patch.object(m,'MAX_PIXELS',10):
            with self.assertRaises(ValueError):m.fetch({'data':self.data})
    def test_private_dns(self):
        for host in ('127.0.0.1','10.0.0.1','169.254.169.254','::1','fe80::1','192.0.2.1'):
            with patch.object(socket,'getaddrinfo',return_value=[(socket.AF_INET,socket.SOCK_STREAM,6,'',(host,80))]):
                with self.assertRaises(ValueError):m._connection_socket(('example.com',80))
    def test_mixed_dns(self):
        with patch.object(socket,'getaddrinfo',return_value=[(2,1,6,'',('8.8.8.8',80)),(2,1,6,'',('127.0.0.1',80))]):
            with self.assertRaises(ValueError):m._connection_socket(('example.com',80))
    def test_literal_connect(self):
        with patch.object(socket,'getaddrinfo',return_value=[(2,1,6,'',('8.8.8.8',443))]),patch.object(socket,'create_connection') as connect:
            m._connection_socket(('example.com',443));self.assertEqual(connect.call_args.args[0],('8.8.8.8',443))
    def test_credential_url(self):
        with self.assertRaises(ValueError):m._url_bytes('https://user:secret@example.com/x')
    def test_animation(self):
        b=io.BytesIO();Image.new('RGB',(3,3)).save(b,format='WEBP',save_all=True,append_images=[Image.new('RGB',(3,3),'red')],duration=100)
        with self.assertRaises(ValueError):m.fetch({'data':base64.b64encode(b.getvalue()).decode()})
if __name__=='__main__':unittest.main()
