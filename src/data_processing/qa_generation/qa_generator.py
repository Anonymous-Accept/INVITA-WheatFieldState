"""
QA Pair Generator for INVITA NeurIPS 2026 Baseline

Generates multiple-choice diagnostic probes for 4 tasks:
- NDVI: NDVI numeric regression (181,555 plot-date examples)
- LAI: LAI numeric regression (17,816 plot-date examples)
- FCover: FCover numeric regression (24,612 plot-date examples)
- Zadoks: Zadoks growth stage numeric regression (3,420 plot-date examples)

Strategy:
- Only multiple-choice questions (A/B/C/D format)
- Based on benchmark-derived examples, no synthetic targets
- Diverse modality combinations (can use subset or all available modalities)
- Different difficulty levels and question types

Question Types:
1. Direct value prediction (30%): "What is the NDVI value?"
2. Temporal reasoning (25%): "How does LAI change over time?"
3. Multimodal comparison (20%): "Which sensor shows higher NDVI?"
4. Range/category (15%): "Is the FCover in the high/medium/low range?"
5. Diagnostic (10%): "Why is growth stage delayed compared to normal?"
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import argparse
import json
import os
import random
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from src.data_processing.loaders.multimodal_aligner import MultimodalAligner


@dataclass
class MultipleChoiceQuestion:
    """A multiple-choice question with 4 options."""
    question_id: str
    task_name: str
    question_type: str  # direct_value, temporal, multimodal_comparison, range, diagnostic
    difficulty: str  # easy, medium, hard
    question_text: str
    options: Dict[str, str]  # {"A": "...", "B": "...", "C": "...", "D": "..."}
    correct_answer: str  # "A", "B", "C", or "D"
    explanation: str
    target_uid: str
    required_modalities: List[str]  # Which modalities are needed to answer
    metadata: Dict[str, Any]  # Additional info (plot_uid, date, etc.)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class QAGenerator:
    """Generate QA pairs for INVITA multimodal tasks."""

    def __init__(self, task_name: str, seed: int = 42):
        """
        Initialize QA generator.

        Args:
            task_name: One of the 4 INVITA tasks
            seed: Random seed for reproducibility
        """
        self.task_name = task_name
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Load data
        print(f"Loading data for {task_name}...")
        self.aligner = MultimodalAligner(task_name=task_name, lazy_load=True)
        self.targets = self.aligner.targets
        self.inputs_index = self.aligner.inputs_index

        # Task-specific configuration
        self.config = self._get_task_config()

        print(f"Loaded {len(self.targets):,} targets")
        print(f"Target range: {self.config['value_range']}")

    def _get_task_config(self) -> Dict[str, Any]:
        """Get task-specific configuration."""
        configs = {
            'NDVI': {
                'target_name': 'NDVI',
                'unit': 'NDVI unit (0-1)',
                'value_range': (0.0, 1.0),
                'value_bins': [(0.0, 0.3, 'low'), (0.3, 0.6, 'medium'), (0.6, 1.0, 'high')],
                'description': 'Normalized Difference Vegetation Index',
            },
            'LAI': {
                'target_name': 'LAI',
                'unit': 'm²/m²',
                'value_range': (0.0, 8.0),  # Filter outliers
                'value_bins': [(0.0, 1.5, 'low'), (1.5, 3.5, 'medium'), (3.5, 8.0, 'high')],
                'description': 'Leaf Area Index',
            },
            'FCover': {
                'target_name': 'Fractional Cover',
                'unit': 'fraction (0-1)',
                'value_range': (0.0, 1.0),
                'value_bins': [(0.0, 0.3, 'sparse'), (0.3, 0.7, 'moderate'), (0.7, 1.0, 'dense')],
                'description': 'Fraction of ground covered by vegetation',
            },
            'Zadoks': {
                'target_name': 'Zadoks Score',
                'unit': 'Zadoks scale (10-99)',
                'value_range': (10, 99),
                'value_bins': [
                    (10, 29, 'vegetative'),
                    (30, 49, 'stem extension'),
                    (50, 69, 'heading'),
                    (70, 89, 'flowering/grain fill'),
                    (90, 99, 'ripening')
                ],
                'description': 'Wheat growth stage on Zadoks scale',
            }
        }
        return configs[self.task_name]

    def generate_questions(
        self,
        num_questions: int = 10000,
        type_distribution: Dict[str, float] = None
    ) -> List[MultipleChoiceQuestion]:
        """
        Generate multiple-choice questions.

        Args:
            num_questions: Number of questions to generate
            type_distribution: Distribution of question types

        Returns:
            List of MultipleChoiceQuestion objects
        """
        if type_distribution is None:
            type_distribution = {
                'direct_value': 0.30,
                'temporal': 0.25,
                'multimodal_comparison': 0.20,
                'range': 0.15,
                'diagnostic': 0.10,
            }

        questions = []
        question_id_counter = 0

        # Calculate number of questions per type
        type_counts = {
            qtype: int(num_questions * prob)
            for qtype, prob in type_distribution.items()
        }

        # Adjust to match exact total
        remaining = num_questions - sum(type_counts.values())
        if remaining > 0:
            type_counts['direct_value'] += remaining

        print(f"\nGenerating {num_questions} questions:")
        for qtype, count in type_counts.items():
            print(f"  {qtype}: {count}")

        # Generate questions by type
        import time
        start_time = time.time()

        for qtype, count in type_counts.items():
            print(f"\n{'='*60}")
            print(f"Generating {count} {qtype} questions...")
            print('='*60)
            type_start = time.time()

            for i in range(count):
                try:
                    if qtype == 'direct_value':
                        q = self._generate_direct_value_question(question_id_counter)
                    elif qtype == 'temporal':
                        q = self._generate_temporal_question(question_id_counter)
                    elif qtype == 'multimodal_comparison':
                        q = self._generate_multimodal_comparison_question(question_id_counter)
                    elif qtype == 'range':
                        q = self._generate_range_question(question_id_counter)
                    elif qtype == 'diagnostic':
                        q = self._generate_diagnostic_question(question_id_counter)
                    else:
                        continue

                    if q:
                        questions.append(q)
                        question_id_counter += 1

                        # Progress update every 500 questions
                        if (i + 1) % 500 == 0:
                            elapsed = time.time() - type_start
                            rate = (i + 1) / elapsed
                            remaining = (count - i - 1) / rate if rate > 0 else 0
                            print(f"  Progress: {i+1}/{count} ({(i+1)/count*100:.1f}%) | "
                                  f"Rate: {rate:.1f} q/s | "
                                  f"ETA: {remaining/60:.1f} min")

                except Exception as e:
                    print(f"  Warning: Failed to generate question: {e}")
                    continue

            type_elapsed = time.time() - type_start
            print(f"  ✓ Completed {qtype}: {len([q for q in questions if q.question_type == qtype])} questions in {type_elapsed/60:.1f} min")

        total_elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✓ Successfully generated {len(questions)} questions")
        print(f"  Total time: {total_elapsed/60:.1f} minutes")
        print(f"  Average rate: {len(questions)/total_elapsed:.2f} questions/second")
        print('='*60)
        return questions

    def _sample_target(self, filters: Dict[str, Any] = None) -> pd.Series:
        """Sample a random target with optional filters."""
        df = self.targets.copy()

        # Apply filters
        if filters:
            for key, value in filters.items():
                if key in df.columns:
                    if isinstance(value, (list, tuple)):
                        df = df[df[key].isin(value)]
                    else:
                        df = df[df[key] == value]

        if len(df) == 0:
            # Fallback to unfiltered sample
            df = self.targets.copy()

        return df.sample(n=1, random_state=random.randint(0, 1000000)).iloc[0]

    def _generate_direct_value_question(self, qid: int) -> Optional[MultipleChoiceQuestion]:
        """Generate a direct value prediction question."""
        target = self._sample_target()
        target_value = target['target_value_num']

        # Skip invalid values
        if pd.isna(target_value):
            return None

        # For LAI, filter outliers
        if self.task_name == 'LAI' and target_value > 8.0:
            return None

        date = target.get('target_date', 'unknown date')
        plot_uid = target.get('plot_uid', 'unknown')

        # Generate question
        question_text = f"Based on the available multimodal observations from {date}, what could be the most likely {self.config['target_name']} value for plot {plot_uid}?"

        # Generate options (correct + 3 distractors)
        correct_value = target_value
        options_values = self._generate_numeric_options(correct_value, self.config['value_range'])

        # Randomly assign to A/B/C/D
        option_letters = ['A', 'B', 'C', 'D']
        random.shuffle(option_letters)

        options = {}
        correct_answer = None
        for i, (letter, value) in enumerate(zip(option_letters, options_values)):
            options[letter] = f"{value:.3f} {self.config['unit']}"
            if abs(value - correct_value) < 0.001:
                correct_answer = letter

        # Get required modalities
        required_modalities = self._get_modalities_for_target(target['target_uid'])

        return MultipleChoiceQuestion(
            question_id=f"{self.task_name}_dv_{qid:06d}",
            task_name=self.task_name,
            question_type='direct_value',
            difficulty='easy',
            question_text=question_text,
            options=options,
            correct_answer=correct_answer,
            explanation=f"The actual measured {self.config['target_name']} value is {correct_value:.3f}.",
            target_uid=target['target_uid'],
            required_modalities=required_modalities,
            metadata={
                'plot_uid': str(plot_uid),
                'date': str(date),
                'instrument': str(target.get('instrument', 'unknown')),
                'true_value': float(correct_value),
            }
        )

    def _generate_temporal_question(self, qid: int) -> Optional[MultipleChoiceQuestion]:
        """Generate a temporal reasoning question."""
        # Find a plot with multiple observations
        plot_counts = self.targets['plot_uid'].value_counts()
        plots_with_multiple = plot_counts[plot_counts >= 2].index.tolist()

        if not plots_with_multiple:
            return None

        selected_plot = random.choice(plots_with_multiple)
        plot_data = self.targets[self.targets['plot_uid'] == selected_plot].sort_values('target_date')

        if len(plot_data) < 2:
            return None

        # Select two time points
        early = plot_data.iloc[0]
        late = plot_data.iloc[-1]

        early_value = early['target_value_num']
        late_value = late['target_value_num']

        if pd.isna(early_value) or pd.isna(late_value):
            return None

        # Calculate change
        change = late_value - early_value
        change_pct = (change / early_value * 100) if early_value != 0 else 0

        # Generate question
        question_text = f"For plot {selected_plot}, the {self.config['target_name']} was {early_value:.3f} on {early['target_date']}. What may be the {self.config['target_name']} value on {late['target_date']}?"

        # Generate options
        options_values = self._generate_numeric_options(late_value, self.config['value_range'])

        option_letters = ['A', 'B', 'C', 'D']
        random.shuffle(option_letters)

        options = {}
        correct_answer = None
        for letter, value in zip(option_letters, options_values):
            options[letter] = f"{value:.3f}"
            if abs(value - late_value) < 0.001:
                correct_answer = letter

        trend = "increased" if change > 0 else "decreased"

        return MultipleChoiceQuestion(
            question_id=f"{self.task_name}_temp_{qid:06d}",
            task_name=self.task_name,
            question_type='temporal',
            difficulty='medium',
            question_text=question_text,
            options=options,
            correct_answer=correct_answer,
            explanation=f"The {self.config['target_name']} {trend} from {early_value:.3f} to {late_value:.3f} (change: {change:+.3f}, {change_pct:+.1f}%).",
            target_uid=late['target_uid'],
            required_modalities=self._get_modalities_for_target(late['target_uid']),
            metadata={
                'plot_uid': str(selected_plot),
                'early_date': str(early['target_date']),
                'late_date': str(late['target_date']),
                'early_value': float(early_value),
                'late_value': float(late_value),
                'change': float(change),
            }
        )

    def _generate_multimodal_comparison_question(self, qid: int) -> Optional[MultipleChoiceQuestion]:
        """Generate a multimodal comparison question."""
        # This works best for NDVI which has greenseeker + UAV
        if self.task_name != 'NDVI':
            # For other tasks, fall back to direct value
            return self._generate_direct_value_question(qid)

        # Find targets with multiple modalities
        target = self._sample_target()
        modalities = self._get_modalities_for_target(target['target_uid'])

        if len(modalities) < 2:
            # Need at least 2 modalities for comparison
            return self._generate_direct_value_question(qid)

        question_text = f"When comparing {modalities[0]} and {modalities[1]} measurements for plot {target['plot_uid']} on {target['target_date']}, which modality combination could provide the most comprehensive vegetation assessment?"

        options = {
            'A': f"Only {modalities[0]} data is sufficient",
            'B': f"Only {modalities[1]} data is sufficient",
            'C': f"Both {modalities[0]} and {modalities[1]} provide complementary information",
            'D': "Neither modality is reliable for this assessment"
        }

        return MultipleChoiceQuestion(
            question_id=f"{self.task_name}_mm_{qid:06d}",
            task_name=self.task_name,
            question_type='multimodal_comparison',
            difficulty='hard',
            question_text=question_text,
            options=options,
            correct_answer='C',
            explanation=f"Multiple modalities provide complementary perspectives: {modalities[0]} and {modalities[1]} capture different spatial and spectral characteristics.",
            target_uid=target['target_uid'],
            required_modalities=modalities,
            metadata={
                'plot_uid': str(target['plot_uid']),
                'date': str(target['target_date']),
                'modalities': modalities,
            }
        )

    def _generate_range_question(self, qid: int) -> Optional[MultipleChoiceQuestion]:
        """Generate a range/category classification question."""
        target = self._sample_target()
        target_value = target['target_value_num']

        if pd.isna(target_value):
            return None

        # Filter outliers for LAI
        if self.task_name == 'LAI' and target_value > 8.0:
            return None

        # Determine the correct category
        correct_category = None
        for min_val, max_val, category in self.config['value_bins']:
            if min_val <= target_value < max_val:
                correct_category = category
                break

        if correct_category is None:
            return None

        # Generate question
        question_text = f"Based on the observed {self.config['target_name']} value of {target_value:.3f} for plot {target['plot_uid']}, which category may best describe the vegetation status?"

        # Create options from all categories
        all_categories = [cat for _, _, cat in self.config['value_bins']]

        # Ensure we have exactly 4 options
        if len(all_categories) == 3:
            all_categories.append(f"extremely {all_categories[-1]}")
        elif len(all_categories) == 5:
            # Use all 5 but pick 4
            pass

        option_letters = ['A', 'B', 'C', 'D']
        random.shuffle(all_categories)

        options = {}
        correct_answer = None
        for letter, category in zip(option_letters, all_categories[:4]):
            options[letter] = category.capitalize()
            if category == correct_category:
                correct_answer = letter

        # If correct_category not in shuffled options, replace one
        if correct_answer is None:
            replace_idx = random.randint(0, 3)
            replace_letter = option_letters[replace_idx]
            options[replace_letter] = correct_category.capitalize()
            correct_answer = replace_letter

        return MultipleChoiceQuestion(
            question_id=f"{self.task_name}_range_{qid:06d}",
            task_name=self.task_name,
            question_type='range',
            difficulty='easy',
            question_text=question_text,
            options=options,
            correct_answer=correct_answer,
            explanation=f"The value {target_value:.3f} falls in the '{correct_category}' range for {self.config['target_name']}.",
            target_uid=target['target_uid'],
            required_modalities=self._get_modalities_for_target(target['target_uid']),
            metadata={
                'plot_uid': str(target['plot_uid']),
                'date': str(target.get('target_date', 'unknown')),
                'true_value': float(target_value),
                'category': correct_category,
            }
        )

    def _generate_diagnostic_question(self, qid: int) -> Optional[MultipleChoiceQuestion]:
        """Generate a diagnostic/reasoning question."""
        target = self._sample_target()
        target_value = target['target_value_num']

        if pd.isna(target_value):
            return None

        # Determine if value is anomalous
        mean_value = self.targets['target_value_num'].mean()
        std_value = self.targets['target_value_num'].std()

        is_low = target_value < (mean_value - std_value)
        is_high = target_value > (mean_value + std_value)

        if is_low:
            question_text = f"The {self.config['target_name']} value for plot {target['plot_uid']} on {target['target_date']} is {target_value:.3f}, which is notably low. What could be the most plausible explanation?"

            options = {
                'A': "Measurement error or sensor malfunction",
                'B': "Early growth stage or recent sowing",
                'C': "Environmental stress (drought, nutrient deficiency)",
                'D': "All of the above may contribute"
            }
            correct_answer = 'D'
            explanation = f"Low {self.config['target_name']} ({target_value:.3f}) may result from multiple factors including growth stage, environmental stress, or measurement issues."

        elif is_high:
            question_text = f"The {self.config['target_name']} value for plot {target['plot_uid']} on {target['target_date']} is {target_value:.3f}, which is notably high. What could be the most likely cause?"

            options = {
                'A': "Peak growth stage with optimal conditions",
                'B': "Dense vegetation cover",
                'C': "Sensor saturation at high biomass",
                'D': "Any of the above may contribute"
            }
            correct_answer = 'D'
            explanation = f"High {self.config['target_name']} ({target_value:.3f}) typically indicates vigorous growth but may also involve sensor limitations."

        else:
            # Normal range - ask about factors
            question_text = f"For plot {target['plot_uid']}, which factor is LEAST likely to significantly affect the {self.config['target_name']} measurement?"

            options = {
                'A': "Time of day during measurement",
                'B': "Crop growth stage",
                'C': "Soil moisture content",
                'D': "Plot identification number"
            }
            correct_answer = 'D'
            explanation = "Plot ID is just a label and doesn't affect the actual measurement, while all other factors may influence vegetation indices."

        return MultipleChoiceQuestion(
            question_id=f"{self.task_name}_diag_{qid:06d}",
            task_name=self.task_name,
            question_type='diagnostic',
            difficulty='hard',
            question_text=question_text,
            options=options,
            correct_answer=correct_answer,
            explanation=explanation,
            target_uid=target['target_uid'],
            required_modalities=self._get_modalities_for_target(target['target_uid']),
            metadata={
                'plot_uid': str(target['plot_uid']),
                'date': str(target.get('target_date', 'unknown')),
                'true_value': float(target_value),
                'mean_value': float(mean_value),
            }
        )

    def _generate_numeric_options(self, correct_value: float, value_range: Tuple[float, float], num_options: int = 4) -> List[float]:
        """Generate numeric options including the correct value and plausible distractors."""
        min_val, max_val = value_range
        options = [correct_value]

        # Generate distractors
        for _ in range(num_options - 1):
            # Add variation (5-20% difference)
            variation = correct_value * random.uniform(0.05, 0.20)
            if random.random() > 0.5:
                distractor = correct_value + variation
            else:
                distractor = correct_value - variation

            # Clamp to valid range
            distractor = max(min_val, min(max_val, distractor))
            options.append(distractor)

        # Sort and return
        options.sort()
        return options

    def _sample_modality_subset(
        self,
        target_uid: str,
        strategy: str = 'mixed'
    ) -> Tuple[List[str], List[str]]:
        """
        Sample a subset of modalities/inputs for a target.

        Args:
            target_uid: Target unique identifier
            strategy: 'all', 'sample', 'single', or 'mixed'

        Returns:
            Tuple of (modality_list, selected_asset_uids)
        """
        # Get all inputs for this target
        inputs = self.inputs_index[self.inputs_index['target_uid'] == target_uid]

        if len(inputs) == 0:
            return ['unknown'], []

        # Get asset info
        asset_uids = inputs['asset_uid'].tolist()
        assets = self.aligner.assets[self.aligner.assets['asset_uid'].isin(asset_uids)]

        # Get all unique modalities
        all_modalities = assets['modality_verified'].unique().tolist()

        # Determine strategy (mixed = 30% all, 50% sample, 20% single)
        if strategy == 'mixed':
            rand = random.random()
            if rand < 0.3:
                strategy = 'all'
            elif rand < 0.8:  # 0.3 + 0.5
                strategy = 'sample'
            else:
                strategy = 'single'

        # Apply strategy
        if strategy == 'all':
            # Use all available modalities and inputs
            return all_modalities, asset_uids

        elif strategy == 'single':
            # Use only one modality type
            selected_modality = random.choice(all_modalities)
            modality_assets = assets[assets['modality_verified'] == selected_modality]

            # Sample 30-70% of inputs from this modality
            n_sample = max(1, int(len(modality_assets) * random.uniform(0.3, 0.7)))
            sampled = modality_assets.sample(n=min(n_sample, len(modality_assets)))

            return [selected_modality], sampled['asset_uid'].tolist()

        else:  # strategy == 'sample'
            # Randomly select 1-3 modality types
            n_modalities = random.randint(1, min(3, len(all_modalities)))
            selected_modalities = random.sample(all_modalities, n_modalities)

            # For each selected modality, sample 30-70% of inputs
            selected_assets = []
            for mod in selected_modalities:
                modality_assets = assets[assets['modality_verified'] == mod]
                n_sample = max(1, int(len(modality_assets) * random.uniform(0.3, 0.7)))
                sampled = modality_assets.sample(n=min(n_sample, len(modality_assets)))
                selected_assets.extend(sampled['asset_uid'].tolist())

            return selected_modalities, selected_assets

    def _get_modalities_for_target(self, target_uid: str, use_sampling: bool = True) -> List[str]:
        """Get list of modalities for a target (with optional sampling)."""
        strategy = 'mixed' if use_sampling else 'all'
        modalities, _ = self._sample_modality_subset(target_uid, strategy=strategy)
        return modalities

    def save_questions(self, questions: List[MultipleChoiceQuestion], output_path: Path):
        """Save questions to JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for q in questions:
                f.write(json.dumps(q.to_dict()) + '\n')

        print(f"\nSaved {len(questions)} questions to {output_path}")

    def close(self):
        """Close aligner."""
        self.aligner.close()


def main():
    """Generate QA pairs for all tasks."""
    tasks = [
        ('NDVI', 12000),
        ('LAI', 10000),
        ('FCover', 10000),
        ('Zadoks', 5000),
    ]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(os.environ.get('INVITA_QA_OUTPUT_ROOT', 'outputs/qa_generation/generated_qa')),
        help='Directory for generated QA JSONL files. Keep this outside committed reviewer files.',
    )
    args = parser.parse_args()
    output_dir = args.output_dir

    for task_name, num_questions in tasks:
        print(f"\n{'='*80}")
        print(f"Generating QA pairs for {task_name}")
        print(f"{'='*80}")

        generator = QAGenerator(task_name=task_name, seed=42)

        try:
            questions = generator.generate_questions(num_questions=num_questions)

            # Save to file
            output_file = output_dir / f"{task_name}_qa.jsonl"
            generator.save_questions(questions, output_file)

            # Print statistics
            print(f"\nStatistics:")
            print(f"  Total questions: {len(questions)}")

            types = {}
            difficulties = {}
            for q in questions:
                types[q.question_type] = types.get(q.question_type, 0) + 1
                difficulties[q.difficulty] = difficulties.get(q.difficulty, 0) + 1

            print(f"  By type:")
            for qtype, count in types.items():
                print(f"    {qtype}: {count}")

            print(f"  By difficulty:")
            for diff, count in difficulties.items():
                print(f"    {diff}: {count}")

        finally:
            generator.close()

    print(f"\n{'='*80}")
    print("QA generation complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
