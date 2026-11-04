"""Post-generation contracts, without a token-level enforcer."""
import unittest
from jsonschema import Draft202012Validator,ValidationError
from umilife.grammar import role_schema

class Contracts(unittest.TestCase):
    def test_planner_rejects_tool_name_as_route(self):
        with self.assertRaises(ValidationError):
            Draft202012Validator(role_schema('planner')).validate({'thought':'test','next':'demo.read'})
    def test_global_caller_requires_action(self):
        value={'thought':'test','next':'caller','tool':'weather.get_forecast'}
        validator=Draft202012Validator(role_schema('global'))
        with self.assertRaises(ValidationError):validator.validate(value)
        validator.validate({**value,'action':{'name':'weather.get_forecast','arguments':{'date':'2026-11-03'}}})
    def test_terminal_and_caller_contracts(self):
        for role,target in (('global',{'thought':'x','next':'give_up'}),('summarizer',{'answer':'x'}),('caller',{'name':'alarm.list','arguments':{}})):
            Draft202012Validator(role_schema(role)).validate(target)

if __name__=='__main__':unittest.main()
