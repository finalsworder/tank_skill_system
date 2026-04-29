from __future__ import annotations
import json

class LLMAgent:
    def __init__(self, api, action_registry, system_prompt=''):
        self.api = api
        self.action_registry = action_registry
        self.system_prompt = system_prompt
        self.history = []

    def reset(self):
        self.history = []

    def select_actions(self, battlefield_summary):
        action_list = self.action_registry.list_actions()
        prompt = (
            self.system_prompt
            + "\n可选行动指令：" + ', '.join(action_list)
            + "\n请输出JSON数组，每个元素包含tank_id、action_id、params。"
            + "\n战场摘要：\n" + battlefield_summary
        )
        text = self.api.chat(prompt, self.history)
        self.history.append({'role': 'assistant', 'content': text})
        return json.loads(text)
