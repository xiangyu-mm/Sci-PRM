python generate_protein.py \
  --stage exec \
  --input-file ./dataset/mol_instruction_step/output_no_ref/protein_function_reasoning.jsonl \
  --output-file ./dataset/mol_instruction_step/output_with_multi_tool/protein_function.jsonl \
  --limit 0 \
  --max-workers 8