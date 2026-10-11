"""What the API accepts. Anything that does not match is rejected with a 422 before our code runs."""
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.security import MAX_PASSWORD, MIN_PASSWORD

_EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s.]{2,}$")


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")  # unknown fields are an error, not silently ignored


def _email(value: str) -> str:
    value = value.strip().lower()
    if not _EMAIL_RE.fullmatch(value):
        raise ValueError("not a valid email address")
    return value


class SignupIn(_Body):
    email: str
    password: str = Field(min_length=MIN_PASSWORD, max_length=MAX_PASSWORD)
    business_name: str = Field(min_length=2, max_length=80)

    _check_email = field_validator("email")(_email)

    @field_validator("business_name")
    @classmethod
    def _name(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 2:
            raise ValueError("business name is too short")
        return v


class LoginIn(_Body):
    email: str
    password: str = Field(min_length=1, max_length=MAX_PASSWORD)

    _check_email = field_validator("email")(_email)


class UseCaseIn(_Body):
    use_case_id: str = Field(min_length=1, max_length=50)


class DataIn(_Body):
    customers: int = Field(default=5000, ge=1000, le=5000)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
