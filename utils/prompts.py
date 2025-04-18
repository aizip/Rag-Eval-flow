
from typing import Dict, List, Optional

# Metric definitions dictionary
# Add any custom metrics here in the form of
# "metric_name": "metric definition"
METRIC_DEFINITIONS = {
    "query relevance": "How much the information in the response is related to the query, and not necessarily how well it answers the query. A high-score indicates that all the information in the response is relevant to the query. A low-score indicates that the information in the response is largely irrelevant. A score of -1 indicates that this metric is irrelevant for this response.",
    "query completeness": "How thoroughly the response answers the query. A high-score indicates that the response answers all parts of the query. A low-score indicates that the response barely answers the query, or does not answer it at all. A response with highly-relevant information but that does not answer the query at all would get a low score.",
    "context adherence": "How much of the response is directly based on the context. A high-score indicates that the response uses information only from the context, even if it does not answer the query at all. A low score indicates that the response does not use information from the context, even if the response accurately answers the query.",
    "context completeness": "How much out of the total relevant context is incorporated into the response. A high score indicates that the response uses nearly all the relevant information from the context, even if the response quality itself is poor. A low score indicates that the response uses little to none of the relevant information from the context, even if the response is accurate and the quality itself is high.",
    "response coherence": "The overall coherence of the response, such as grammar, fluency and readability. A high score indicates that the response is coherent and easy to read. A low score indicates that the response is incoherent, difficult to read, or contains grammatical errors. This metric should not take into account the content of the response, only the fluency and readability.",
    "response length": "The length of the response, including whether it is too short or too long.",
    "refusal quality": "How well the response explains its refusal to answer the query. A high score indicates that the refusal is well-explained and justified. A low score indicates that the refusal is poorly explained or unjustified. A score of -1 indicates that this metric is irrelevant for this response. This score should be outputted only when the response does not refuse to respond.",
    "refusal clarification quality": "How likely the clarification questions asked during refusal are to lead to a successful response. A high score indicates that the clarification questions are likely to lead to a successful response. A low score indicates that the clarification questions are unlikely to lead to a successful response. A score of -1 indicates that this metric is irrelevant for this response. This score should be outputted only when the response does not refuse to respond.",
}


def get_metric_definition(metric_name: str) -> str:
    """Get the definition for a specific metric."""
    assert metric_name in METRIC_DEFINITIONS, (
        f"Metric name {metric_name} is not valid. Choose from {list(METRIC_DEFINITIONS.keys())}."
    )
    return METRIC_DEFINITIONS[metric_name]


def get_available_metrics() -> List[str]:
    """Get list of all available metrics."""
    return list(METRIC_DEFINITIONS.keys())


def filter_metrics(categories: Optional[List[str]] = None) -> List[str]:
    """
    Filter metrics by category prefix.

    Args:
        categories: List of category prefixes (e.g., ["query", "response"])
                   If None, returns all metrics

    Returns:
        List of metric names matching the categories
    """
    if categories is None:
        return get_available_metrics()

    result = []
    for metric in METRIC_DEFINITIONS:
        if any(metric.startswith(category) for category in categories):
            result.append(metric)
    return result

def format_rag_prompt(query: str, context: str, accepts_sys: bool) -> str:
    system_prompt = """
    You are a helpful assistant that provides answers to queries based on the provided context.

    You MUST clearly refuse to answer the query and ask for additional information from the user if the answer cannot be found in the context.
    The output should not contain your judgment on answerability, only your answer OR refusal + clarifications.

    Stay within the bounds of the provided context and avoid making assumptions.


    """
    user_prompt = f"""

    # Role and Task Description
    Judge if the following query is answerable from ONLY the provided context.
    If so, provide a complete, grounded answer to the query, and do not mention your judgement.
    Try to address all aspects of the query, but if certain parts are not answerable, clearly state that you do not have enough information.

    OTHERWISE, refuse clearly to answer and ask for the additional information you require from the user. 
    You should give a concise explanation of why you cannot answer the query based on the context, and ask for more relevant information from the user.

    # Task
    Given the following query and context, please provide your response:
    Query: {query}

    Context: {context}

    WITHOUT mentioning your judgement either your grounded answer, OR refusal and clarifications:
    """

    messages = (
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if accepts_sys
        else [{"role": "user", "content": system_prompt + user_prompt}]
    )
    return messages

def _format_judge_system_prompt(
    metric_name: str,
    metric_definition: str,
) -> str:
    assert metric_name in get_available_metrics(), (
        f"Metric name {metric_name} is not valid. Choose from {get_available_metrics()}."
    )

    system_prompt_template = f"""
    # Role and Objective
    You are an expert at evaluating SLM-generated responses according to the following metric: {metric_name.upper()}.
    
    ## Metric Definition
    {metric_name.upper()} is defined as follows: {metric_definition}

    # Evaluation Instructions
    You will be provided a query, a context, and a response. 
    Your task is to evaluate the RESPONSE based on the provided QUERY and CONTEXT, using the metric defined above.
    Follow the guidance provided in the metric definition to evaluate the response.

    Be as objective as possible in your evaluation. Do not allow the response's conclusions to influence your own evaluations.

    Output your evaluation according to the JSON format specified.
    """
    return system_prompt_template


def _format_judge_user_prompt(
    metric_name: str, query: str, context: List[str], response: str
) -> str:
    user_prompt_template = f"""
    # Task Description
    Your job is to evaluate the quality of a LLM-generated response based on the provided metric.
    The metric you are evaluating is: {metric_name.upper()}.

    # Evaluation Outputs
    Your output should be twofold: 1) a numerical score, and 2) a short explanation of the score.

    The score should be a number between 0 and 10, following the criteria outlined in the metric definition.
    The explanation should be a short sentence explaining the score.
    
    ## Output Format
    Please provide your evaluation according to the following JSON format:
    {{
        "score": <score>, 
        "explanation": "<explanation>"
    }}
    
    # Do below
    Evaluate the following RESPONSE, given the following QUERY and CONTEXT. 
    QUERY: {query}
    CONTEXT: {context}
    RESPONSE: {response}

    """
    return user_prompt_template


def format_messages(
    query: str, context: List[str], response: str, metric_name: str
) -> List[Dict[str, str]]:
    """
    Format the messages for the OpenAI API call.

    Args:
        query: The user's query
        context: List of context snippets
        response: The LLM's response
        metric_name: Which metric to evaluate (must be in METRICS)

    Returns:
        List of message dictionaries for the OpenAI API
    """
    assert metric_name in METRIC_DEFINITIONS, (
        f"Metric name {metric_name} is not valid. Choose from {list(METRIC_DEFINITIONS.keys())}."
    )

    metric_definition = METRIC_DEFINITIONS[metric_name]
    system_prompt = _format_judge_system_prompt(metric_name, metric_definition)
    user_prompt = _format_judge_user_prompt(metric_name, query, context, response)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return messages


def format_batch_request_task(
    metric_name: str, eval_datapoint: Dict, generation_args: Dict
):
    """
    Format a batch request task for the OpenAI API.
    Args:
        metric_name: The name of the metric to evaluate
        eval_data: Dictionary containing:
            - id: Sample ID of the datapoint
            - messages: List of messages to be sent to the OpenAI API
        generation_args: Dictionary containing generation arguments
            (e.g., model, temperature, max_tokens, messages, etc.)
    Returns:
        A dictionary representing the task for the OpenAI API
    """
    task = {
        "custom_id": f"{metric_name}_{eval_datapoint['id']}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": generation_args["model"],
            "temperature": generation_args["temperature"],
            "max_tokens": generation_args["max_tokens"],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "evaluation",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "score": {
                                "type": "number",
                                "description": "The score retrieved from the evaluation.",
                            },
                            "explanation": {
                                "type": "string",
                                "description": "A detailed explanation of the evaluation result.",
                            },
                        },
                        "required": ["score", "explanation"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            "messages": eval_datapoint["messages"],
        },
    }
    return task

def format_judge_answerability_prompt(self, query:str, context: list):
    """
    Format the prompt for the model to judge answerability.

    Args:
        query: The input query
        context: The context to check against

    Returns:
        Formatted prompt string
    """
    system_prompt = """
    You are a helpful assistant that provides answers to queries based on the provided context.

    You should refuse to answer the query and ask for additional information from the user if the answer cannot be found in the context.
    Output only your response.

    """
    user_prompt = f"""
    Judge if the following query is answerable from ONLY the provided context.
    Do not rely on any external knowledge. Use only the context provided in your decision.

    Output only a boolean value (True or False) and a short explanation.
    Query: {query}

    Context: {context}

    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages