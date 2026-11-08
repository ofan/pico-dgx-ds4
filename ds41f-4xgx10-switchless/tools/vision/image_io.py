"""Bounded image ingress. Connections use the validated literal IP, not a second DNS lookup."""
import base64
import http.client
import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

MAX_BYTES = 32 << 20
MAX_REQUEST_BYTES = 64 << 20
MAX_PIXELS = 40_000_000
TYPES = {'image/png': 'PNG', 'image/jpeg': 'JPEG', 'image/webp': 'WEBP'}

def _connection_socket(address, timeout=20, source_address=None):
    host, port = address
    answers = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not answers: raise ValueError('image host has no addresses')
    if any(not ipaddress.ip_address(a[4][0]).is_global or ipaddress.ip_address(a[4][0]).is_multicast for a in answers):
        raise ValueError('image URL destination is not public')
    # No hostname is given to create_connection: DNS cannot rebind after this check.
    return socket.create_connection((answers[0][4][0], port), timeout, source_address)

def _url_bytes(url):
    for redirect in range(4):
        u = urlsplit(url)
        if u.scheme not in ('http', 'https') or not u.hostname or u.username is not None or u.password is not None:
            raise ValueError('only public HTTP(S) image URLs without credentials are supported')
        conn_cls = http.client.HTTPSConnection if u.scheme == 'https' else http.client.HTTPConnection
        conn = conn_cls(u.hostname, u.port, timeout=20)
        conn._create_connection = _connection_socket
        try:
            conn.request('GET', (u.path or '/') + ('?'+u.query if u.query else ''), headers={'User-Agent':'pico-vision/1','Accept-Encoding':'identity'})
            r = conn.getresponse()
            if r.status in (301,302,303,307,308):
                loc = r.getheader('Location')
                if redirect == 3 or not loc: raise ValueError('too many or invalid image redirects')
                url = urljoin(url, loc);continue
            if r.status != 200: raise ValueError('image URL returned non-200 status')
            if r.getheader('Content-Encoding', 'identity') != 'identity': raise ValueError('encoded image responses are unsupported')
            mime = r.getheader('Content-Type','').split(';')[0].strip().lower()
            if mime not in TYPES: raise ValueError('unsupported image MIME type')
            n = r.getheader('Content-Length')
            if n is not None and (int(n)<1 or int(n)>MAX_BYTES): raise ValueError('image body exceeds limit')
            raw=r.read(MAX_BYTES+1)
            if len(raw)>MAX_BYTES:raise ValueError('image body exceeds limit')
            return raw,mime
        finally:conn.close()
    raise ValueError('redirect limit')

def fetch(record):
    if not isinstance(record,dict):raise ValueError('image record must be an object')
    url=record.get('url');data=record.get('data');mime=None
    src=record.get('source')
    if isinstance(src,dict):
        if src.get('url'):url=src['url']
        elif src.get('type')=='base64':data=src.get('data');mime=src.get('media_type')
    if isinstance(url,str) and url.startswith(('http://','https://')):
        raw,mime=_url_bytes(url)
    else:
        if isinstance(url,str) and url.startswith('data:'):
            meta,sep,data=url.partition(',')
            if not sep or not meta.endswith(';base64'):raise ValueError('image data URL must be base64')
            mime=meta[5:-7].lower()
        elif url is not None:raise ValueError('image URL must be data: or public HTTP(S); no file paths')
        if not isinstance(data,str) or len(data)>4*((MAX_BYTES+2)//3):raise ValueError('invalid or oversized image base64')
        if mime is not None and mime not in TYPES:raise ValueError('unsupported image MIME type')
        try:raw=base64.b64decode(data,validate=True)
        except Exception:raise ValueError('invalid image base64') from None
    if not raw or len(raw)>MAX_BYTES:raise ValueError('empty or oversized image')
    # Decode limits apply to URL and data: input alike, before the publisher's processor allocates patches.
    from PIL import Image
    import io, warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error',Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(raw)) as im:
            if im.width*im.height>MAX_PIXELS or im.width<1 or im.height<1:raise ValueError('image pixel limit')
            if im.format not in TYPES.values() or (mime is not None and TYPES[mime]!=im.format):raise ValueError('image MIME does not match bytes')
            if getattr(im,'n_frames',1)!=1:raise ValueError('animated images are unsupported')
            im.verify()
    return raw
