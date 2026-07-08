"""Core models — 用户角色扩展。"""

from django.db import models
from django.contrib.auth.models import User


class UserProfile(models.Model):
    """用户角色扩展。"""

    ROLE_CHOICES = [
        ("admin", "系统管理员"),
        ("legal_lead", "法律主管"),
        ("legal_user", "法律员工"),
        ("hr_lead", "人事主管"),
        ("hr_user", "人事员工"),
        ("general_user", "员工"),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    role = models.CharField(max_length=32, choices=ROLE_CHOICES, default="general_user")

    class Meta:
        verbose_name = "User Profile"
        verbose_name_plural = "User Profiles"

    def __str__(self):
        return f"{self.user.username} — {self.get_role_display()}"

    @property
    def is_admin(self):
        return self.role == "admin"

    @property
    def is_legal(self):
        return self.role in ("legal_lead", "legal_user")

    @property
    def is_hr(self):
        return self.role in ("hr_lead", "hr_user")

    @property
    def is_lead(self):
        return self.role in ("admin", "legal_lead", "hr_lead")

    @property
    def domain_filter(self):
        """该角色能访问的知识库领域。"""
        if self.role == "admin":
            return None  # 全部
        if self.is_legal:
            return "law"
        if self.is_hr:
            return "general"
        return "general"
