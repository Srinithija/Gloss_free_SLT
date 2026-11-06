"""Start the Streamlit app.

Python 3.9's ssl loads the whole Windows certificate store in one call, and a single malformed
certificate on this machine makes that call fail (ASN1 NOT_ENOUGH_DATA) when Tornado/Streamlit
import. This launcher loads the store certificate-by-certificate, skipping unparsable entries.
"""
import os
import ssl
import sys
from pathlib import Path

# transformers would otherwise import the env's TensorFlow (via is_tf_tensor during decoding), whose old
# generated protobuf modules are incompatible with the protobuf 4.x that Streamlit needs (in app_deps).
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"


def _load_windows_store_certs_tolerant(self, storename, purpose):
    for cert, encoding, trust in ssl.enum_certificates(storename):
        if encoding == "x509_asn" and (trust is True or purpose.oid in trust):
            try:
                self.load_verify_locations(cadata=cert)
            except ssl.SSLError:
                pass


if sys.platform == "win32":
    ssl.SSLContext._load_windows_store_certs = _load_windows_store_certs_tolerant

if __name__ == "__main__":
    from streamlit.web import cli

    app = Path(__file__).with_name("app.py")
    # Streamlit assumes development mode when not installed under site-packages (it lives in app_deps)
    sys.argv = ["streamlit", "run", str(app), "--global.developmentMode", "false", *sys.argv[1:]]
    sys.exit(cli.main())
