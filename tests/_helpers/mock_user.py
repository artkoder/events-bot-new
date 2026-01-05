
from dataclasses import dataclass

@dataclass
class MockUser:
    user_id: int
    is_superadmin: bool = False
