
import os
import unittest
from unittest.mock import patch, MagicMock
from tests._helpers.mock_user import MockUser

# We import from main directly. 
# Note: main.py has a lot of top-level code, so we might need to mock some things to just test a function.
# But has_admin_access and is_e2e_tester seem self-contained enough.

from main import has_admin_access, is_e2e_tester

class TestSupertesterRole(unittest.TestCase):
    def setUp(self):
        # Clear env vars before each test
        os.environ.pop("DEV_MODE", None)
        os.environ.pop("E2E_TESTER_ID", None)

    def test_is_e2e_tester_no_dev_mode(self):
        os.environ["DEV_MODE"] = "0"
        os.environ["E2E_TESTER_ID"] = "12345"
        self.assertFalse(is_e2e_tester(12345))

    def test_is_e2e_tester_dev_mode_correct_id(self):
        os.environ["DEV_MODE"] = "1"
        os.environ["E2E_TESTER_ID"] = "12345"
        self.assertTrue(is_e2e_tester(12345))

    def test_is_e2e_tester_dev_mode_wrong_id(self):
        os.environ["DEV_MODE"] = "1"
        os.environ["E2E_TESTER_ID"] = "12345"
        self.assertFalse(is_e2e_tester(67890))

    def test_is_e2e_tester_no_id_env(self):
        os.environ["DEV_MODE"] = "1"
        self.assertFalse(is_e2e_tester(12345))

    def test_has_admin_access_superadmin(self):
        user = MockUser(user_id=999, is_superadmin=True)
        self.assertTrue(has_admin_access(user))

    def test_has_admin_access_e2e_tester_dev_mode(self):
        os.environ["DEV_MODE"] = "1"
        os.environ["E2E_TESTER_ID"] = "12345"
        user = MockUser(user_id=12345, is_superadmin=False)
        self.assertTrue(has_admin_access(user))

    def test_has_admin_access_non_admin_no_dev(self):
        os.environ["DEV_MODE"] = "0"
        os.environ["E2E_TESTER_ID"] = "12345"
        user = MockUser(user_id=12345, is_superadmin=False)
        self.assertFalse(has_admin_access(user))

if __name__ == "__main__":
    unittest.main()
