from __future__ import annotations
class GLM4API:
    def __init__(self, client, model='glm-4'): self.client=client; self.model=model
    def chat(self, content, history=None):
        messages=[] if history is None else list(history); messages.append({'role':'user','content':content}); response=self.client.chat.completions.create(model=self.model,messages=messages); return response.choices[0].message.content.strip()
