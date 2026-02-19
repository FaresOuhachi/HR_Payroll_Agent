"""
Payroll Tools — Functions the Agent Can Call
=============================================================================
CONCEPT: Tool Calling

Tools are Python functions that the LLM agent can invoke to perform actions.
The LLM doesn't execute code directly — instead, it:
  1. Sees a list of available tools with descriptions
  2. Decides which tool(s) to call based on the user's request
  3. Generates the arguments for the tool call
  4. The system executes the tool and returns results to the LLM
  5. The LLM uses the results to form its response

The @tool decorator from LangChain:
  - Registers the function as an available tool
  - Uses the docstring as the tool description (sent to the LLM)
  - Uses type hints for argument validation
  - The LLM sees: tool name, description, and parameter types

WHY tools instead of having the LLM do math?
  LLMs are notoriously bad at precise arithmetic. Tools ensure:
  - Exact calculations (no hallucinated numbers)
  - Real database lookups (not invented data)
  - Auditable operations (every tool call is logged)
=============================================================================
"""

from langchain_core.tools import tool

from src.db.engine import async_session_maker
from src.db import repositories as repo


@tool
async def get_employee_info(employee_code: str) -> dict:
    """
    Get detailed information about an employee by their employee code (e.g., EMP001).
    Returns: name, department, position, salary info, benefits info, and tax info.
    Use this tool when you need to look up an employee's details before calculating payroll.
    """
    async with async_session_maker() as db:
        employee = await repo.get_employee_by_code(db, employee_code)
        if not employee:
            return {"error": f"Employee {employee_code} not found"}

        return {
            "employee_code": employee.employee_code,
            "full_name": employee.full_name,
            "email": employee.email,
            "department": employee.department,
            "position": employee.position,
            "salary_info": employee.salary_info,
            "tax_info": employee.tax_info,
            "benefits_info": employee.benefits_info,
            "is_active": employee.is_active,
        }


@tool
async def calculate_gross_pay(employee_code: str, period: str = "monthly") -> dict:
    """
    Calculate the gross pay for an employee for a given period.
    Args:
        employee_code: The employee's code (e.g., EMP001)
        period: The pay period - "monthly" or "annual"
    Returns: gross pay amount with calculation breakdown.
    """
    async with async_session_maker() as db:
        employee = await repo.get_employee_by_code(db, employee_code)
        if not employee:
            return {"error": f"Employee {employee_code} not found"}

        annual_salary = employee.salary_info.get("annual_salary", 0)

        if period == "monthly":
            gross_pay = annual_salary / 12
        else:
            gross_pay = annual_salary

        return {
            "employee_code": employee_code,
            "employee_name": employee.full_name,
            "period": period,
            "annual_salary": annual_salary,
            "gross_pay": round(gross_pay, 2),
            "currency": employee.salary_info.get("currency", "USD"),
        }


@tool
async def calculate_deductions(employee_code: str, gross_pay: float) -> dict:
    """
    Calculate all deductions (tax, insurance, retirement) for an employee.
    Args:
        employee_code: The employee's code (e.g., EMP001)
        gross_pay: The gross pay amount to calculate deductions from
    Returns: itemized deductions and net pay.
    """
    async with async_session_maker() as db:
        employee = await repo.get_employee_by_code(db, employee_code)
        if not employee:
            return {"error": f"Employee {employee_code} not found"}

        tax_bracket = employee.tax_info.get("tax_bracket", 0) / 100
        health_insurance = employee.benefits_info.get("health_insurance_monthly", 0)
        retirement_pct = employee.benefits_info.get("retirement_pct", 0) / 100

        # Calculate each deduction
        tax_amount = round(gross_pay * tax_bracket, 2)
        retirement_amount = round(gross_pay * retirement_pct, 2)
        total_deductions = round(tax_amount + health_insurance + retirement_amount, 2)
        net_pay = round(gross_pay - total_deductions, 2)

        return {
            "employee_code": employee_code,
            "employee_name": employee.full_name,
            "gross_pay": gross_pay,
            "deductions": {
                "tax": {"rate": f"{tax_bracket*100}%", "amount": tax_amount},
                "health_insurance": {"amount": health_insurance},
                "retirement": {"rate": f"{retirement_pct*100}%", "amount": retirement_amount},
            },
            "total_deductions": total_deductions,
            "net_pay": net_pay,
        }


@tool
async def calculate_net_pay(employee_code: str) -> dict:
    """
    Calculate the complete monthly net pay for an employee (gross - all deductions).
    This is an all-in-one tool that does the full payroll calculation.
    Args:
        employee_code: The employee's code (e.g., EMP001)
    Returns: Complete pay breakdown including gross, all deductions, and net pay.
    """
    async with async_session_maker() as db:
        employee = await repo.get_employee_by_code(db, employee_code)
        if not employee:
            return {"error": f"Employee {employee_code} not found"}

        # Gross pay
        annual_salary = employee.salary_info.get("annual_salary", 0)
        monthly_gross = annual_salary / 12

        # Deductions
        tax_bracket = employee.tax_info.get("tax_bracket", 0) / 100
        health_insurance = employee.benefits_info.get("health_insurance_monthly", 0)
        retirement_pct = employee.benefits_info.get("retirement_pct", 0) / 100

        tax_amount = round(monthly_gross * tax_bracket, 2)
        retirement_amount = round(monthly_gross * retirement_pct, 2)
        total_deductions = round(tax_amount + health_insurance + retirement_amount, 2)
        net_pay = round(monthly_gross - total_deductions, 2)

        return {
            "employee_code": employee_code,
            "employee_name": employee.full_name,
            "period": "monthly",
            "gross_pay": round(monthly_gross, 2),
            "deductions": {
                "income_tax": {"rate": f"{tax_bracket*100}%", "amount": tax_amount},
                "health_insurance": {"monthly": health_insurance},
                "retirement": {"rate": f"{retirement_pct*100}%", "amount": retirement_amount},
            },
            "total_deductions": total_deductions,
            "net_pay": net_pay,
            "currency": employee.salary_info.get("currency", "USD"),
        }


@tool
async def get_leave_balance(employee_code: str) -> dict:
    """
    Check an employee's remaining PTO (Paid Time Off) balance.
    Args:
        employee_code: The employee's code (e.g., EMP001)
    Returns: PTO days total, used, and remaining.
    """
    async with async_session_maker() as db:
        employee = await repo.get_employee_by_code(db, employee_code)
        if not employee:
            return {"error": f"Employee {employee_code} not found"}

        total = employee.benefits_info.get("pto_days_total", 0)
        used = employee.benefits_info.get("pto_days_used", 0)
        remaining = total - used

        return {
            "employee_code": employee_code,
            "employee_name": employee.full_name,
            "pto_days_total": total,
            "pto_days_used": used,
            "pto_days_remaining": remaining,
        }


@tool
async def search_employees_by_department(department: str) -> dict:
    """
    Search for all employees in a specific department.
    Args:
        department: The department name (e.g., "Engineering", "Finance", "Human Resources", "Sales")
    Returns: List of employees in that department with their basic info.
    """
    async with async_session_maker() as db:
        employees = await repo.list_employees(db, department=department)
        return {
            "department": department,
            "employee_count": len(employees),
            "employees": [
                {
                    "employee_code": e.employee_code,
                    "full_name": e.full_name,
                    "position": e.position,
                    "annual_salary": e.salary_info.get("annual_salary", 0),
                }
                for e in employees
            ],
        }


@tool
async def calculate_department_payroll(department: str) -> dict:
    """
    Calculate the total monthly payroll cost for an entire department.
    Args:
        department: The department name (e.g., "Engineering")
    Returns: Total gross, total deductions, and total net pay for the department.
    """
    async with async_session_maker() as db:
        employees = await repo.list_employees(db, department=department)
        if not employees:
            return {"error": f"No employees found in department '{department}'"}

        total_gross = 0
        total_net = 0
        total_deductions = 0
        employee_details = []

        for emp in employees:
            monthly_gross = emp.salary_info.get("annual_salary", 0) / 12
            tax = monthly_gross * emp.tax_info.get("tax_bracket", 0) / 100
            insurance = emp.benefits_info.get("health_insurance_monthly", 0)
            retirement = monthly_gross * emp.benefits_info.get("retirement_pct", 0) / 100
            deductions = tax + insurance + retirement
            net = monthly_gross - deductions

            total_gross += monthly_gross
            total_deductions += deductions
            total_net += net

            employee_details.append({
                "employee_code": emp.employee_code,
                "name": emp.full_name,
                "gross": round(monthly_gross, 2),
                "net": round(net, 2),
            })

        return {
            "department": department,
            "employee_count": len(employees),
            "total_monthly_gross": round(total_gross, 2),
            "total_monthly_deductions": round(total_deductions, 2),
            "total_monthly_net": round(total_net, 2),
            "employees": employee_details,
        }


# Registry of all payroll tools
PAYROLL_TOOLS = [
    get_employee_info,
    calculate_gross_pay,
    calculate_deductions,
    calculate_net_pay,
    get_leave_balance,
    search_employees_by_department,
    calculate_department_payroll,
]
