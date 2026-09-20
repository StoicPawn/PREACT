from __future__ import annotations

import os
import uvicorn

if __name__=="__main__":
    uvicorn.run(
        "preact.platform.api:app",
        host=os.getenv("PREACT_HISTORICAL_API_HOST","127.0.0.1"),
        port=int(os.getenv("PREACT_HISTORICAL_API_PORT","8790")),
        workers=1,
    )
