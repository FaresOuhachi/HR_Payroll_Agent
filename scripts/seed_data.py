"""
Seed Data Script — Populate the database with sample data
=============================================================================
CONCEPT: Data Seeding

Seeding fills the database with realistic test data so you can:
  1. Test API endpoints without manual data entry
  2. Demo the system with meaningful examples
  3. Develop agent tools against real-looking data

This script creates:
  - 3 users (admin, manager, employee) for testing auth & RBAC
  - 10 employees with realistic HR data
  - Sample data modeled after an Algerian HR tech company (Vane LLC)

Run: python -m scripts.seed_data
=============================================================================
"""

import asyncio
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from passlib.context import CryptContext
from sqlalchemy import text

from src.db.engine import async_session_maker, engine
from src.db.models import User, Employee


# Password hashing context
# CONCEPT: Never store plain text passwords! bcrypt is a one-way hash function.
# When a user logs in, we hash their input and compare to the stored hash.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


async def seed_users(session):
    """Create sample users with different roles."""
    # Check if users already exist
    result = await session.execute(text("SELECT COUNT(*) FROM users"))
    count = result.scalar()
    if count > 0:
        print(f"  Users table already has {count} records, skipping...")
        return

    users = [
        User(
            username="admin",
            hashed_password=pwd_context.hash("admin123"),
            role="admin",
            full_name="System Administrator",
            email="admin@faresouhachi.com",
        ),
        User(
            username="manager",
            hashed_password=pwd_context.hash("manager123"),
            role="manager",
            full_name="Fatima Zerhouni",
            email="fatima.manager@faresouhachi.com",
        ),
        User(
            username="employee",
            hashed_password=pwd_context.hash("employee123"),
            role="employee",
            full_name="Amina Benali",
            email="amina.user@faresouhachi.com",
        ),
    ]

    session.add_all(users)
    await session.commit()
    print(f"  Created {len(users)} users (admin/manager/employee)")


async def seed_employees(session):
    """Load employee data from JSON and insert into database."""
    # Check if employees already exist
    result = await session.execute(text("SELECT COUNT(*) FROM employees"))
    count = result.scalar()
    if count > 0:
        print(f"  Employees table already has {count} records, skipping...")
        return

    # Load sample data
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "sample_data",
        "employees.json",
    )

    with open(data_path) as f:
        employees_data = json.load(f)

    employees = []
    for emp_data in employees_data:
        employee = Employee(
            employee_code=emp_data["employee_code"],
            full_name=emp_data["full_name"],
            email=emp_data["email"],
            department=emp_data["department"],
            position=emp_data["position"],
            salary_info=emp_data["salary_info"],
            tax_info=emp_data["tax_info"],
            benefits_info=emp_data["benefits_info"],
        )
        employees.append(employee)

    session.add_all(employees)
    await session.commit()
    print(f"  Created {len(employees)} employees")


async def main():
    """Run all seed functions."""
    print("Seeding database...")
    print("=" * 50)

    async with async_session_maker() as session:
        print("\n1. Seeding users...")
        await seed_users(session)

        print("\n2. Seeding employees...")
        await seed_employees(session)

    print("\n" + "=" * 50)
    print("Seeding complete!")
    print("\nTest credentials:")
    print("  admin    / admin123    (role: admin)")
    print("  manager  / manager123  (role: manager)")
    print("  employee / employee123 (role: employee)")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
