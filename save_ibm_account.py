"""
Save IBM Quantum credentials for qiskit-ibm-runtime, so that the other
scripts can call QiskitRuntimeService() without arguments.

Set two environment variables and run the script once:

    export IBM_QUANTUM_TOKEN="<your API key>"
    export IBM_QUANTUM_INSTANCE="<your instance CRN>"
    python save_ibm_account.py

The account is stored in ~/.qiskit/qiskit-ibm.json. Nothing is submitted.
"""

import os
import sys

from qiskit_ibm_runtime import QiskitRuntimeService


def main() -> None:
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    instance = os.environ.get("IBM_QUANTUM_INSTANCE")
    if not token or not instance:
        sys.exit("Set IBM_QUANTUM_TOKEN and IBM_QUANTUM_INSTANCE first.")
    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform",
        token=token,
        instance=instance,
        overwrite=True,
    )
    print("Saved the IBM Quantum account.")


if __name__ == "__main__":
    main()
