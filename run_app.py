"""Launch the ViT translator demo on http://localhost:8502 (port 8502, so it can run beside the old app on 8501).

    env\\python.exe run_app.py [--server.port 8502 ...]
"""
import os
import ssl
import sys


def _windows_store_as_pem(self, storename, purpose):
    pem = [ssl.DER_cert_to_PEM_cert(c) for c, enc, trust in ssl.enum_certificates(storename)
           if enc == 'x509_asn' and (trust is True or purpose.oid in trust)]
    if pem:
        self.load_verify_locations(cadata=''.join(pem))


if __name__ == '__main__':
    if sys.platform == 'win32':
        try:
            ssl.create_default_context()
        except ssl.SSLError:  # some Python/OpenSSL builds reject the Windows store as DER; it loads as PEM
            ssl.SSLContext._load_windows_store_certs = _windows_store_as_pem
    from streamlit.web import cli
    app = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'slt_vit', 'app.py')
    sys.argv = ['streamlit', 'run', app, '--server.address', 'localhost', '--server.port', '8502',
                '--browser.gatherUsageStats', 'false', *sys.argv[1:]]
    sys.exit(cli.main())
