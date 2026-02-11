# Prompt Templates

## episode_summary.txt

```
You are summarizing a conversation between a user and an AI assistant.

<conversation>
{transcript}
</conversation>

Write a concise episode summary (200-400 words) that captures:
1. The main topics discussed
2. Key decisions, insights, or conclusions reached
3. Any action items or commitments made
4. The emotional tone and context of the conversation

Write in third person. Focus on information that would be useful for long-term memory.
```

## monthly_rollup.txt

```
You are synthesizing multiple conversation summaries from {month} {year} into a monthly overview.

<episode_summaries>
{episodes}
</episode_summaries>

Write a monthly rollup (300-600 words) that captures:
1. Major themes and recurring topics this month
2. How the user's thinking evolved across conversations
3. Key decisions and their context
4. Important facts, preferences, or relationships mentioned

Prioritize information that builds a coherent picture of this period. Deduplicate across episodes.
```

## topical_rollup.txt

```
You are synthesizing conversation summaries related to the topic: {topic}

<episode_summaries>
{episodes}
</episode_summaries>

Write a topical synthesis (300-600 words) that captures:
1. Everything the user has discussed about {topic}
2. How their views or understanding evolved over time
3. Key facts, opinions, and decisions related to {topic}
4. Connections to other topics or areas of their life

Write as a coherent narrative, not a list of conversations.
```

## core_memory.txt

```
You are synthesizing all available information into a core memory document for an AI agent.

Budget: {context_budget} tokens maximum.

<rollup_summaries>
{rollups}
</rollup_summaries>

Create a structured core memory that includes:
1. **Identity**: Who is this person? Background, profession, key relationships.
2. **Current focus**: What are they working on right now? What matters most?
3. **Preferences & style**: Communication preferences, technical opinions, values.
4. **Key history**: Major life events, career arc, important decisions.
5. **Active threads**: Ongoing projects, unresolved questions, things to follow up on.

This will be placed directly in an AI agent's context window. Be specific and factual. Every sentence should be useful. Stay within the token budget.
```
