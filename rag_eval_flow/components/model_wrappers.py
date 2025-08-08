import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from utils.prompts import format_rag_prompt 

# THIS ONLY WORKS FOR HF MODELS RN, TODO: FIX
def is_system_prompt_supported(tokenizer: AutoTokenizer) -> bool:
    """
    Determines if the tokenizer's chat template supports system prompts.
    It tries to render a test conversation and checks for explicit markers.
    """
    if tokenizer.chat_template:
        accepts_system_prompt = False  
        try:
            test_conversation = [
                {"role": "system", "content": "Test system message."},
                {"role": "user", "content": "Test user message."}
            ]
            tokenizer.apply_chat_template(
                test_conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            accepts_system_prompt = True
            print("Chat template successfully processed a conversation with a system message.")
        except Exception as e:
            print(f"Failed to process chat template with a system message: {e}. Assuming system prompt not directly supported by this template.")
            # accepts_system_prompt remains False

        # Gemma workaround
        if accepts_system_prompt and \
           "system role not supported" in str(tokenizer.chat_template).lower():
            print("Warning: Template processed system message, but template string contains 'system role not supported'. "
                  "Prioritizing template string hint and disabling distinct system prompt.")
            accepts_system_prompt = False
        
        print(f"Model accepts system prompt (based on template analysis): {accepts_system_prompt}")
        return accepts_system_prompt
    else:
        print("Tokenizer has no chat_template. `format_rag_prompt` will handle system prompt formatting (e.g., by prepending to user message).")
        # If no template, format_rag_prompt needs to know it should include the system prompt.
        return True

class BaseModelWrapper(ABC):
    def __init__(self, model_name_or_path: str, model_type: str, max_new_tokens: int = 512, 
                 batch_size: int = 16, device: str = "auto", model_kwargs: Optional[Dict] = None, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.model_type = model_type
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.model_kwargs = model_kwargs if model_kwargs else {}
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.tokenizer = None
        self.model = None
        print(f"BaseModelWrapper initialized for {model_name_or_path} on device '{self.device}' with batch size {self.batch_size}")

    @abstractmethod
    def load_model_and_tokenizer(self):
        pass

    @abstractmethod
    def generate_answers(self, query_texts: List[str], documents: List[List[str] | str]) -> List[str]:
        pass

# TODO: support more backends (llama.cpp, vllm, bitnet.cpp, etc.)
class LlamaCppModelWrapper(BaseModelWrapper):
    def __init__(self, model_name_or_path: str, model_type: str, 
                 tokenizer_name_or_path: Optional[str] = None, 
                 lora_adapter_path: Optional[str] = None, **kwargs):
        try:
            # Import llama_cpp and its chat handler for prompt formatting.
            import llama_cpp
            from llama_cpp.llama_chat_format import LlamaChatCompletionHandler
        except ImportError as e:
            print("Could not import llama-cpp-python. Check your installation.")
            raise
        super().__init__(model_name_or_path, model_type, **kwargs)
        self.tokenizer_name_or_path = tokenizer_name_or_path if tokenizer_name_or_path else self.model_name_or_path
        import os
        if ".gguf" not in os.path.splitext(self.model_name_or_path)[-1]:
            print("Warning: LlamaCpp backend selected without base model gguf. Make sure model_path is set correctly.")
        self.lora_adapter_path = lora_adapter_path
        if self.lora_adapter_path and ".gguf" not in os.path.splitext(self.lora_adapter_path)[-1]:
            print("Warning: LlamaCpp backend selected without LoRA adapter gguf. Make sure lora_adapter_path is set correctly.")
        
        # The model object, will be initialized in load_model_and_tokenizer.
        self.model: Optional[llama_cpp.Llama] = None
        # most modern chat-finetuned GGUF models support system prompts.
        # TODO add check
        self._model_accepts_system_prompt = True 
        self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        """
        Loads the GGUF model and prepares it for inference.
        This version sets batching parameters for efficient processing.
        """
        print(f"Attempting to load {self.model_name_or_path} using the llama-cpp-python backend.")
        
        # Create a dictionary of arguments to pass to the Llama constructor.
        model_kwargs = self.model_kwargs.copy()

        # Handle GPU offloading.
        if self.device != 'cpu':
            model_kwargs['n_gpu_layers'] = -1
            print("Attempting to use GPU. Offloading all layers.")
        else:
            model_kwargs['n_gpu_layers'] = 0
            print("Running on CPU.")
            
        # Set batching parameters for optimal performance.
        # n_batch: Number of tokens to process in parallel.
        # n_ubatch: Number of sequences to process in parallel.
        ## TODO: Change this! make it discoverable!
        model_kwargs.setdefault('n_batch', 512) 
        model_kwargs.setdefault('n_ubatch', self.batch_size)
        print(f"Batching enabled: n_batch={model_kwargs['n_batch']}, n_ubatch={model_kwargs['n_ubatch']}")


        # Set the lora adapter if provided.
        if self.lora_adapter_path:
            model_kwargs['lora_path'] = self.lora_adapter_path
            print(f"LoRA adapter found at: {self.lora_adapter_path}")

        # Set verbose to False unless explicitly requested.
        model_kwargs.setdefault('verbose', False)

        try:
            self.model = llama_cpp.Llama(
                model_path=self.model_name_or_path,
                **model_kwargs
            )
            # Ensure the model has a chat handler, which is required for formatting prompts.
            if not self.model.chat_handler:
                raise ValueError("Model does not have a chat handler. Batch processing requires a chat-formatted model.")
            
            print(f"Model {self.model_name_or_path} loaded successfully.")
        except Exception as e:
            print(f"Error loading model {self.model_name_or_path} with llama-cpp-python: {e}")
            raise


    def generate_answers(self, query_texts: List[str], documents: List[List[str] | str]) -> List[str]:
        """
        Generates answers for a batch of queries using the model's batched inference capability.
        """
        if not self.model or not self.model.chat_handler:
            raise RuntimeError("Model and its chat handler must be loaded before generating answers.")
        
        generated_texts = []
        if len(query_texts) != len(documents):
            raise ValueError("Length of query_texts and documents must be the same.")

        # batched inference
        for i in tqdm(range(0, len(query_texts), self.batch_size), desc=f"Generating answers with {self.model_name_or_path}"):
            batch_queries = query_texts[i:i + self.batch_size]
            batch_documents = documents[i:i + self.batch_size]
            
            batch_prompts = []
            for query, doc_list in zip(batch_queries, batch_documents):
                current_docs = doc_list if isinstance(doc_list, (list, str)) else str(doc_list)
                
                # fixed prompts for now, TODO: add modular prompt
                formatted_prompt_messages = format_rag_prompt(
                    query=query, 
                    context=current_docs,
                    tokenizer_chat_template=None,
                    model_accepts_system_prompt=self._model_accepts_system_prompt
                )

                string_prompt = self.model.chat_handler.apply_chat_template(
                    formatted_prompt_messages, 
                    add_generation_prompt=True
                )
                batch_prompts.append(string_prompt)

            # Print the first prompt for debugging and format checking.
            if i == 0:
                print("First formatted prompt to be sent to the model:")
                print(batch_prompts[0])

            # `create_completion` accepts a list of strings for batched inference.
            outputs = self.model.create_completion(
                prompt=batch_prompts,
                max_tokens=self.max_new_tokens,
            )
            

            batch_generated_texts = [choice['text'].strip() for choice in outputs['choices']]
            generated_texts.extend(batch_generated_texts)

            # Sanity check: print the first generated answer of the first batch.
            if i == 0:
                print("SANITY CHECK: FIRST SLM RESPONSE:")
                print("====="*10)
                print(generated_texts[0])
                
        return generated_texts



class HuggingFaceModelWrapper(BaseModelWrapper):
    def __init__(self, model_name_or_path: str, model_type: str, 
                 tokenizer_name_or_path: Optional[str] = None, 
                 lora_adapter_path: Optional[str] = None, **kwargs):
        super().__init__(model_name_or_path, model_type, **kwargs)
        self.tokenizer_name_or_path = tokenizer_name_or_path if tokenizer_name_or_path else model_name_or_path
        self.lora_adapter_path = lora_adapter_path
        self._model_accepts_system_prompt = True # default, will be checked after tokenizer is loaded
        self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        print(f"Attempting to load {self.model_name_or_path} using the transformers backend.")
        print(f"Loading tokenizer: {self.tokenizer_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Tokenizer pad_token was None, set to eos_token.")

        # Determine if the model/tokenizer's chat template supports system prompts.
        self._model_accepts_system_prompt = is_system_prompt_supported(self.tokenizer)


        print(f"Loading base model: {self.model_name_or_path} of type: {self.model_type}")
        model_class = AutoModelForCausalLM if self.model_type == "causal_lm" else AutoModelForSeq2SeqLM
        
        # Add device_map to model_kwargs if not already set by user, and device is not cpu
        current_model_kwargs = self.model_kwargs.copy()
        print(current_model_kwargs)

        # args dict preprocessing, some of these require torch objects
        if self.device != 'cpu' and 'device_map' not in current_model_kwargs:
            current_model_kwargs['device_map'] = self.device
        
        if "torch_dtype" in current_model_kwargs.keys():
            if current_model_kwargs["torch_dtype"] == 'bfloat16':
                current_model_kwargs["torch_dtype"] = torch.bfloat16
        
        try:
            base_model = model_class.from_pretrained(
                self.model_name_or_path,
                **current_model_kwargs
            )
            if self.model_type == "causal_lm" and hasattr(base_model.config, 'pad_token_id') and base_model.config.pad_token_id is None:
                 if self.tokenizer.pad_token_id is not None:
                    base_model.config.pad_token_id = self.tokenizer.pad_token_id
                    print(f"Set base_model.config.pad_token_id to tokenizer.pad_token_id: {self.tokenizer.pad_token_id}")

            if self.lora_adapter_path:
                print(f"Loading and applying LoRA adapter from: {self.lora_adapter_path}")
                from peft import PeftModel 
                self.model = PeftModel.from_pretrained(base_model, self.lora_adapter_path)
                # Optional: self.model = self.model.merge_and_unload()
                print("LoRA adapter applied.")
            else:
                self.model = base_model
            
            if self.device == 'cpu': # If device_map was not used, explicitly move to CPU if needed
                self.model.to(self.device)

            self.model.eval() 
            print(f"Model {self.model_name_or_path} loaded successfully on {self.model.device} with args:")
            print(self.model.config)

        except Exception as e:
            print(f"Error loading model {self.model_name_or_path}: {e}")
            if "attn_implementation" in str(e) and "flash_attention_2" in str(e) and self.model_type == "causal_lm":
                print("Failed with flash_attention_2. Retrying without explicit attn_implementation.")
                current_model_kwargs.pop("attn_implementation")
                base_model = model_class.from_pretrained(
                    self.model_name_or_path,
                    **current_model_kwargs
                )
                if self.lora_adapter_path:
                    from peft import PeftModel
                    self.model = PeftModel.from_pretrained(base_model, self.lora_adapter_path)
                else:
                    self.model = base_model
                if self.device == 'cpu': self.model.to(self.device)
                self.model.eval()
                print(f"Model {self.model_name_or_path} loaded successfully on {self.model.device} (without flash_attention_2).")

            else:
                raise


    def generate_answers(self, query_texts: List[str], documents: List[List[str] | str]) -> List[str]:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded before generating answers.")
        
        generated_texts = []
        if len(query_texts) != len(documents):
            raise ValueError("Length of query_texts and documents must be the same.")

        for i in tqdm(range(0, len(query_texts), self.batch_size), desc=f"Generating answers with {self.model_name_or_path}"):
            batch_queries = query_texts[i:i + self.batch_size]
            batch_documents = documents[i:i + self.batch_size]

            batch_input_messages = []
            for query, doc_list in zip(batch_queries, batch_documents):
                # format_rag_prompt expects a list of context strings or a single context string
                current_docs = doc_list if isinstance(doc_list, (list, str)) else str(doc_list)
                formatted_prompt_messages = format_rag_prompt(
                    query=query, 
                    context=current_docs,
                    tokenizer_chat_template=self.tokenizer.chat_template,
                    model_accepts_system_prompt=self._model_accepts_system_prompt
                )
                batch_input_messages.append(formatted_prompt_messages)
            
            # Print first prompt to check for format issues:
            if i == 0:
                print("====="*10)
                print("SANITY CHECK: FIRST USER PROMPT:")
                print("====="*10)
                print(batch_input_messages[0][-1]["content"])

            try: # Tokenize time
                inputs = self.tokenizer.apply_chat_template(
                    batch_input_messages,
                    return_tensors="pt",
                    return_dict=True, # force BatchEncoding return
                    padding="longest",
                    tokenize = True,
                    # max_length can be taken from tokenizer.model_max_length or a config
                    # max_length=self.tokenizer.model_max_length if hasattr(self.tokenizer, 'model_max_length') else 2048,
                    add_generation_prompt=True, # Important for some models
                ).to(self.model.device)
            except Exception as e:
                print(f"Error during tokenization with apply_chat_template: {e}")
                # Hard crash for now, consistent chat template is essential for SLM performance
                # TODO find a more graceful way to tokenize if apply_chat_template fails
                raise

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    # Common generation parameters, TODO: test overridable in yaml
                    # do_sample=False, # for deterministic output
                )

            # Decode 
            if self.model_type == "causal_lm":
                decoded_batch_outputs = []
                for j in range(outputs.shape[0]): # batch
                    input_length = inputs.input_ids.shape[-1]
                    output_tokens = outputs[j][input_length:] # remove input tokens
                    decoded_output = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                    decoded_batch_outputs.append(decoded_output)
                generated_texts.extend(decoded_batch_outputs)
            else: # seq2seq
                decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                generated_texts.extend(decoded_outputs)

            # sanity check output, should correspond to sanity check input
            if i == 0:
                print("SANITY CHECK: FIRST SLM RESPONSE:")
                print("====="*10)
                print(generated_texts[0])
                
        return generated_texts

class HuggingFaceCausalLM(HuggingFaceModelWrapper):
    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(model_name_or_path, **kwargs)

class HuggingFaceSeq2SeqLM(HuggingFaceModelWrapper):
    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(model_name_or_path, **kwargs)

