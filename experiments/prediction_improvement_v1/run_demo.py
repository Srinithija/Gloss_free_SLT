"""Launch the improved demo on http://localhost:8503 (the original app keeps port 8502).

    env\\python.exe experiments\\prediction_improvement_v1\\run_demo.py
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
    app = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo_app.py')
    sys.argv = ['streamlit', 'run', app, '--server.address', 'localhost', '--server.port', '8503',
                '--browser.gatherUsageStats', 'false', *sys.argv[1:]]
    sys.exit(cli.main())
