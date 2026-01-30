"""
Prompting utilities for multiple-choice QA.

Implements prompt templates and zero-shot/few-shot prompting strategies.
"""

import torch
from torch import Tensor
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from part3.nn_utils import softmax


class PromptTemplate:
    """
    Template for formatting QA prompts.
    
    Supports different prompt styles:
    - basic: Simple question-answer format
    - instruction: With explicit instructions
    - few_shot: With example demonstrations
    """
    
    TEMPLATES = {
        "basic": """Context: {context}

Question: {question}

Choices:
{choices_formatted}

Answer:""",
        
        "instruction": """Read the following passage and answer the question by selecting the correct choice.

Passage: {context}

Question: {question}

{choices_formatted}

Select the letter (A, B, C, or D) of the correct answer:""",
        
        "cot": """Let's think step by step to answer this question.

Context: {context}

Question: {question}

Choices:
{choices_formatted}

Let me analyze each option:""",
        
        "simple": """{context}
{question}
{choices_formatted}
The answer is""",
    }
    
    def __init__(
        self,
        template_name: str = "basic",
        custom_template: Optional[str] = None,
        choice_format: str = "letter",  # "letter" (A, B, C, D) or "number" (1, 2, 3, 4)
    ):
        """
        Initialize prompt template.
        
        Args:
            template_name: Name of built-in template to use
            custom_template: Optional custom template string
            choice_format: Format for choice labels ("letter" or "number")
        """
        if custom_template:
            self.template = custom_template
        elif template_name in self.TEMPLATES:
            self.template = self.TEMPLATES[template_name]
        else:
            raise ValueError(f"Unknown template: {template_name}. Available: {list(self.TEMPLATES.keys())}")
        
        self.choice_format = choice_format
    
    def _format_choices(self, choices: List[str]) -> str:
        """Format choices with labels."""
        if self.choice_format == "letter":
            labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
        else:
            labels = [str(i + 1) for i in range(len(choices))]
        
        formatted = []
        for label, choice in zip(labels, choices):
            formatted.append(f"{label}. {choice}")
        
        return "\n".join(formatted)
    
    def format(
        self,
        context: str,
        question: str,
        choices: List[str],
        **kwargs,
    ) -> str:
        """
        Format a single QA example as a prompt.
        
        Args:
            context: Context passage
            question: Question text
            choices: List of answer choices
            **kwargs: Additional template variables
        
        Returns:
            Formatted prompt string
        """
        choices_formatted = self._format_choices(choices)
        
        return self.template.format(
            context=context,
            question=question,
            choices_formatted=choices_formatted,
            **kwargs,
        )
    
    def format_with_answer(
        self,
        context: str,
        question: str,
        choices: List[str],
        answer_idx: int,
    ) -> str:
        """
        Format a QA example with the correct answer (for few-shot examples).
        
        Args:
            context: Context passage
            question: Question text
            choices: List of answer choices
            answer_idx: Index of correct answer
        
        Returns:
            Formatted prompt with answer
        """
        prompt = self.format(context, question, choices)
        
        if self.choice_format == "letter":
            answer_label = chr(ord('A') + answer_idx)
        else:
            answer_label = str(answer_idx + 1)
        
        return f"{prompt} {answer_label}"


class PromptingPipeline:
    """
    Pipeline for prompting-based multiple-choice QA.
    
    Uses the language model to predict the correct answer by computing
    the probability of each choice token given the prompt.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        template: Optional[PromptTemplate] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize prompting pipeline.
        
        Args:
            model: TransformerLM model
            tokenizer: Tokenizer instance
            template: Prompt template (default: basic)
            device: Device to use
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.template = template or PromptTemplate("basic")
        self.device = device
        
        # Get token IDs for choice labels
        self._setup_choice_tokens()
    
    def _setup_choice_tokens(self):
        """Set up token IDs for answer choices."""
        # Common choice tokens to check
        self.choice_tokens = {}
        
        for label in ["A", "B", "C", "D", "a", "b", "c", "d", "1", "2", "3", "4"]:
            # Try with and without space prefix
            for prefix in ["", " "]:
                text = prefix + label
                token_ids = self.tokenizer.encode(text)
                if token_ids:
                    # Use the last token (in case of space)
                    self.choice_tokens[label.upper()] = token_ids[-1]
                    break
    
    @torch.no_grad()
    def predict_single(
        self,
        context: str,
        question: str,
        choices: List[str],
        return_probs: bool = False,
    ) -> int | tuple:
        """
        Predict the correct choice for a single example.
        
        Uses the probability of each choice token given the prompt.
        
        Args:
            context: Context passage
            question: Question text
            choices: List of answer choices
            return_probs: Whether to also return probabilities
        
        Returns:
            Predicted choice index (0-indexed), or tuple (prediction, probs) if return_probs
        """
        self.model.eval()
        
        # TODO: Implement zero-shot prediction
        # Step 1: Format prompt using self.template.format(context, question, choices)
        # Step 2: Tokenize: input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        # Step 3: Get logits for next token: logits = self.model(input_ids)[:, -1, :]
        # Step 4: Get choice labels ["A", "B", "C", "D"][:len(choices)]
        # Step 5: For each label, get token_id from self.choice_tokens[label] and extract logits[0, token_id].item()
        #         If label not in self.choice_tokens, use float("-inf")
        # Step 6: Convert choice_logits to tensor and compute probs = softmax(choice_logits, dim=-1)
        # Step 7: Get prediction = probs.argmax().item()
        # Step 8: Return prediction (and probs.tolist() if return_probs=True)
        
        raise NotImplementedError("Implement predict_single")
    
    @torch.no_grad()
    def predict_batch(
        self,
        examples: List[Dict[str, Any]],
        batch_size: int = 8,
    ) -> List[int]:
        """
        Predict choices for a batch of examples.
        
        Args:
            examples: List of dicts with context, question, choices
            batch_size: Batch size for processing
        
        Returns:
            List of predicted choice indices
        """
        predictions = []
        
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            
            for example in batch:
                pred = self.predict_single(
                    example["context"],
                    example["question"],
                    example["choices"],
                )
                predictions.append(pred)
        
        return predictions
    
    def few_shot_predict(
        self,
        context: str,
        question: str,
        choices: List[str],
        few_shot_examples: List[Dict[str, Any]],
    ) -> int:
        """
        Predict using few-shot prompting.
        
        Args:
            context: Context passage for the test example
            question: Question for the test example
            choices: Choices for the test example
            few_shot_examples: List of demonstration examples with answers
        
        Returns:
            Predicted choice index
        """
        # Build few-shot prompt
        prompt_parts = ["Here are some examples:\n"]
        
        for idx, example in enumerate(few_shot_examples):
            demo = self.template.format_with_answer(
                example["context"],
                example["question"],
                example["choices"],
                example["answer"],
            )
            prompt_parts.append(f"Example {idx + 1}:\n{demo}\n")
        
        prompt_parts.append("\nNow answer this question:\n")
        prompt_parts.append(self.template.format(context, question, choices))
        
        full_prompt = "\n".join(prompt_parts)
        
        # Tokenize and predict
        self.model.eval()
        input_ids = torch.tensor([self.tokenizer.encode(full_prompt)], device=self.device)
        
        # Get logits for the next token
        logits = self.model(input_ids)[:, -1, :]
        
        # Get probabilities for choice tokens
        choice_labels = ["A", "B", "C", "D"][:len(choices)]
        choice_logits = []
        
        for label in choice_labels:
            if label in self.choice_tokens:
                token_id = self.choice_tokens[label]
                choice_logits.append(logits[0, token_id].item())
            else:
                choice_logits.append(float("-inf"))
        
        choice_logits = torch.tensor(choice_logits)
        probs = softmax(choice_logits, dim=-1)
        
        return probs.argmax().item()


def evaluate_prompting(
    pipeline: PromptingPipeline,
    examples: List[Dict[str, Any]],
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    Evaluate prompting pipeline on a dataset.
    
    Args:
        pipeline: PromptingPipeline instance
        examples: List of QA examples with context, question, choices, answer
        batch_size: Batch size
    
    Returns:
        Dictionary with accuracy and predictions
    """
    predictions = pipeline.predict_batch(examples, batch_size)
    
    # Calculate accuracy
    correct = 0
    total = 0
    
    for pred, example in zip(predictions, examples):
        label = example.get("answer", -1)
        if label >= 0:
            total += 1
            if pred == label:
                correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "predictions": predictions,
    }
