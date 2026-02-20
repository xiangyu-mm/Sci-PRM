python generate_protein.py \
  --stage llm \
  --prompt-type no_ref \
  --input-file ./dataset/mol_instruction/Protein-oriented_Instructions/protein_function.json \
  --output-file ./dataset/mol_instruction_step/output_no_ref/protein_function_reasoning.jsonl \
  --model doubao-seed-1-6-lite-251015 \
  --max-workers 8