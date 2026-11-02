import importlib.util
import json
import unittest
from umilife.grammar import role_schema


@unittest.skipUnless(importlib.util.find_spec("lmformatenforcer"),"Install runtime requirements for JSON grammar checks")
class Grammar(unittest.TestCase):
    def parser_after(self,role,text):
        from lmformatenforcer import JsonSchemaParser
        parser=JsonSchemaParser(role_schema(role))
        for char in text:
            self.assertIn(char,parser.get_allowed_characters(),text)
            parser=parser.add_character(char)
        return parser

    def test_tool_name_cannot_replace_planner_route(self):
        parser=self.parser_after("planner",'{"thought":"test","next":"')
        self.assertNotIn("d",parser.get_allowed_characters())
        self.assertIn("c",parser.get_allowed_characters())

    def test_global_caller_requires_an_action(self):
        prefix='{"thought":"test","next":"caller","tool":"weather.get_forecast"'
        parser=self.parser_after("global",prefix)
        self.assertNotIn("}",parser.get_allowed_characters())
        valid=prefix+',"action":{"name":"weather.get_forecast","arguments":{"date":"2026-11-03"}}}'
        self.assertTrue(self.parser_after("global",valid).can_end())

    def test_terminal_and_caller_contracts(self):
        for role,target in (("global",{"thought":"x","next":"give_up"}),
                            ("summarizer",{"answer":"x"}),
                            ("caller",{"name":"alarm.list","arguments":{}})):
            self.assertTrue(self.parser_after(role,json.dumps(target)).can_end())


if __name__=="__main__":
    unittest.main()
