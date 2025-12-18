"""
Core inference engine for the sanity check experiment.
Implements the three experimental conditions from sanitycheckplan.md
"""
import torch
import sys
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import json
from datetime import datetime

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_model

# Use single-shot noise injector specific to sanity check
# This injects noise EXACTLY ONCE at first forward pass (per protocol Section 5C, 6)
from sanitycheck.embedding_noise_single import SingleShotEmbeddingNoise
from sanitycheck.generation_utils import generate_text


@dataclass
class GenerationResult:
    """Stores a single generation result with metadata."""
    condition: str  # "A", "B", or "C"
    prompt_idx: int
    sample_idx: int
    prompt_text: str
    generated_text: str
    timestamp: str
    seed: int = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def save_to_file(self, filepath: Path) -> None:
        """Save individual result to text file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Condition {self.condition}\n")
            f.write(f"# Prompt {self.prompt_idx}, Sample {self.sample_idx}\n")
            f.write(f"# Timestamp: {self.timestamp}\n")
            if self.seed is not None:
                f.write(f"# Seed: {self.seed}\n")
            f.write(f"\n{'='*80}\n")
            f.write(f"PROMPT:\n{self.prompt_text}\n")
            f.write(f"\n{'='*80}\n")
            f.write(f"GENERATED OUTPUT:\n{self.generated_text}\n")


class SanityCheckInference:
    """
    Orchestrates the three experimental conditions:
    - Condition A: Deterministic baseline (temp=0, no noise)
    - Condition B: Temperature sampling (temp>0, no noise)
    - Condition C: Embedding noise (temp=0, with noise)
    """
    
    def __init__(self, config_module, prompts_list: List[str]):
        """
        Args:
            config_module: Configuration module with settings
            prompts_list: List of prompt strings to use
        """
        self.config = config_module
        self.prompts = prompts_list
        self.model = None
        self.tokenizer = None
        self.noise_injector = None
        self.results = []
        
    def setup(self) -> None:
        """Load model and prepare noise injector."""
        print("\n" + "="*80)
        print("SANITY CHECK INFERENCE SETUP")
        print("="*80)
        
        # Load model
        print(f"\nLoading model: {self.config.MODEL_NAME}")
        torch_dtype = getattr(torch, self.config.TORCH_DTYPE)
        self.model, self.tokenizer = load_model(
            model_name=self.config.MODEL_NAME,
            device_map=self.config.DEVICE_MAP,
            torch_dtype=torch_dtype,
            load_in_8bit=self.config.LOAD_IN_8BIT,
        )
        
        # Setup SINGLE-SHOT noise injector for Condition C
        # Critical: This injects noise EXACTLY ONCE at first forward pass (per protocol)
        print(f"\nSetting up SINGLE-SHOT embedding noise injector")
        print(f"  Sigma scale: {self.config.SIGMA_SCALE}")
        print(f"  Noise scope: {self.config.NOISE_SCOPE}")
        print(f"  [Protocol §5C, §6: Noise injected exactly once at first forward pass]")
        
        self.noise_injector = SingleShotEmbeddingNoise(
            sigma_scale=self.config.SIGMA_SCALE,
            noise_scope=self.config.NOISE_SCOPE,
        )
        self.noise_injector.attach_to_model(self.model)
        
        print("\n✓ Setup complete")
        
    def run_condition_a(self) -> List[GenerationResult]:
        """
        Condition A: Deterministic Baseline
        - Temperature = 0 (greedy decoding)
        - No sampling randomness
        - No noise
        - One generation per prompt
        """
        print("\n" + "="*80)
        print("CONDITION A: DETERMINISTIC BASELINE")
        print("="*80)
        print("Settings: temperature=0, no noise, 1 sample per prompt")
        
        results = []
        
        for prompt_idx, prompt_text in enumerate(self.prompts):
            if self.config.SHOW_PROGRESS:
                print(f"\nPrompt {prompt_idx + 1}/{len(self.prompts)}")
            
            # Ensure noise is OFF
            self.noise_injector.deactivate()
            
            # Generate
            generated = generate_text(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt_text,
                do_sample=False,
                temperature=None,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                min_new_tokens=self.config.MIN_NEW_TOKENS,
            )
            
            result = GenerationResult(
                condition="A",
                prompt_idx=prompt_idx,
                sample_idx=0,
                prompt_text=prompt_text,
                generated_text=generated,
                timestamp=datetime.now().isoformat(),
            )
            results.append(result)
            
            if self.config.VERBOSE:
                print(f"  Generated {len(generated)} characters")
        
        print(f"\n✓ Condition A complete: {len(results)} outputs")
        return results
    
    def run_condition_b(self) -> List[GenerationResult]:
        """
        Condition B: Temperature Sampling Baseline
        - Temperature > 0
        - No embedding noise
        - k generations per prompt
        """
        print("\n" + "="*80)
        print("CONDITION B: TEMPERATURE SAMPLING")
        print("="*80)
        print(f"Settings: temperature={self.config.TEMPERATURE}, no noise, {self.config.K_SAMPLES} samples per prompt")
        
        results = []
        
        for prompt_idx, prompt_text in enumerate(self.prompts):
            if self.config.SHOW_PROGRESS:
                print(f"\nPrompt {prompt_idx + 1}/{len(self.prompts)}")
            
            # Ensure noise is OFF
            self.noise_injector.deactivate()
            
            for sample_idx in range(self.config.K_SAMPLES):
                if self.config.VERBOSE and self.config.SHOW_PROGRESS:
                    print(f"  Sample {sample_idx + 1}/{self.config.K_SAMPLES}...", end=" ")
                
                generated = generate_text(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt_text,
                    do_sample=True,
                    temperature=self.config.TEMPERATURE,
                    max_new_tokens=self.config.MAX_NEW_TOKENS,
                    min_new_tokens=self.config.MIN_NEW_TOKENS,
                    # Make sampling reproducible per sample_idx while staying within Condition B
                    seed=sample_idx,
                )
                
                result = GenerationResult(
                    condition="B",
                    prompt_idx=prompt_idx,
                    sample_idx=sample_idx,
                    prompt_text=prompt_text,
                    generated_text=generated,
                    timestamp=datetime.now().isoformat(),
                )
                results.append(result)
                
                if self.config.VERBOSE and self.config.SHOW_PROGRESS:
                    print(f"{len(generated)} chars")
        
        print(f"\n✓ Condition B complete: {len(results)} outputs")
        return results
    
    def run_condition_c(self) -> List[GenerationResult]:
        """
        Condition C: Embedding-Level Noise Intervention
        - Temperature = 0 (deterministic decoding)
        - Embedding noise injection at first forward pass
        - k different noise seeds per prompt
        """
        print("\n" + "="*80)
        print("CONDITION C: EMBEDDING NOISE")
        print("="*80)
        print(f"Settings: temperature=0, embedding noise, {self.config.K_SAMPLES} samples per prompt")
        print(f"  Sigma scale: {self.config.SIGMA_SCALE}")
        print(f"  Noise scope: {self.config.NOISE_SCOPE}")
        
        results = []
        
        for prompt_idx, prompt_text in enumerate(self.prompts):
            if self.config.SHOW_PROGRESS:
                print(f"\nPrompt {prompt_idx + 1}/{len(self.prompts)}")
            
            for sample_idx in range(self.config.K_SAMPLES):
                if self.config.VERBOSE and self.config.SHOW_PROGRESS:
                    print(f"  Sample {sample_idx + 1}/{self.config.K_SAMPLES} (seed={sample_idx})...", end=" ")
                
                # Set unique seed and activate noise
                self.noise_injector.set_seed(sample_idx)
                self.noise_injector.activate()
                
                generated = generate_text(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt_text,
                    do_sample=False,
                    temperature=None,
                    max_new_tokens=self.config.MAX_NEW_TOKENS,
                    min_new_tokens=self.config.MIN_NEW_TOKENS,
                )
                
                # Deactivate noise after generation
                self.noise_injector.deactivate()
                
                result = GenerationResult(
                    condition="C",
                    prompt_idx=prompt_idx,
                    sample_idx=sample_idx,
                    prompt_text=prompt_text,
                    generated_text=generated,
                    timestamp=datetime.now().isoformat(),
                    seed=sample_idx,
                )
                results.append(result)
                
                if self.config.VERBOSE and self.config.SHOW_PROGRESS:
                    print(f"{len(generated)} chars")
        
        print(f"\n✓ Condition C complete: {len(results)} outputs")
        return results
    
    def run_all_conditions(self) -> List[GenerationResult]:
        """Run all three conditions in sequence."""
        print("\n" + "="*80)
        print("RUNNING ALL CONDITIONS")
        print("="*80)
        print(f"Total prompts: {len(self.prompts)}")
        print(f"Expected outputs:")
        print(f"  Condition A: {len(self.prompts)} (1 per prompt)")
        print(f"  Condition B: {len(self.prompts) * self.config.K_SAMPLES}")
        print(f"  Condition C: {len(self.prompts) * self.config.K_SAMPLES}")
        print(f"  Total: {len(self.prompts) * (1 + 2 * self.config.K_SAMPLES)}")
        
        all_results = []
        
        # Run conditions
        all_results.extend(self.run_condition_a())
        all_results.extend(self.run_condition_b())
        all_results.extend(self.run_condition_c())
        
        self.results = all_results
        return all_results
    
    def save_results(self, output_dir: Path) -> None:
        """Save all results to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)
        print(f"Output directory: {output_dir}")
        
        # Save individual files
        if self.config.SAVE_INDIVIDUAL_FILES:
            print("\nSaving individual output files...")
            for result in self.results:
                filename = f"{result.condition}_{result.prompt_idx}_{result.sample_idx}.txt"
                filepath = output_dir / filename
                result.save_to_file(filepath)
            print(f"  Saved {len(self.results)} individual files")
        
        # Save consolidated JSON
        if self.config.SAVE_SUMMARY_JSON:
            print("\nSaving consolidated JSON...")
            json_path = output_dir / "all_results.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'metadata': {
                            'model': self.config.MODEL_NAME,
                            'n_prompts': len(self.prompts),
                            'k_samples': self.config.K_SAMPLES,
                            'temperature': self.config.TEMPERATURE,
                            'sigma_scale': self.config.SIGMA_SCALE,
                            'noise_scope': self.config.NOISE_SCOPE,
                            'total_outputs': len(self.results),
                        },
                        'results': [r.to_dict() for r in self.results],
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"  Saved {json_path}")
        
        print("\n✓ All results saved")

