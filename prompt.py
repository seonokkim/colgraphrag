# =============================================================================
# LLM prompt templates.
# - Prompt bodies stay English for model input; comments above each constant describe phase and purpose.
# - Phase 2–5 and auxiliary inference prompts (images/tables, etc.) live together—read alongside call sites.
# =============================================================================

# --- Phase 2 (pattern.py): small graph pattern from question (entity-type list + ##-separated relation templates) ---
GRAPH_PATTERN_PROMPT = r'''-Goal-
Given a question, generate a compact graph pattern for downstream relation extraction.

-Steps-
1. Identify a small set of likely entity types relevant to the question.
2. Identify possible relation patterns connecting these entity types.
3. Return output in the exact required format:
   - First block: a Python-style list of entity types
   - Then one or more relation pattern blocks separated by `##`
   - End each relation block with `<|COMPLETE|>`
4. Do not add explanation text.

Output format:
["TYPE_1", "TYPE_2", ...]##TYPE_1 <|> relation_name <|> TYPE_2<|COMPLETE|>##TYPE_2 <|> relation_name <|> TYPE_3<|COMPLETE|>

######################
-Real Data-
######################
Question: {question}
Output:'''

# --- Phase 3 (extraction.py): extract entity / relationship records from body text using Phase 2 pattern ---
EXTRACT_RELATION_PROMPT = r'''-Goal-
Extract entities and relations from input text following the provided graph pattern guidance.

-Output format-
Use `##` to separate multiple records.

Entity record format (MUST start with literal "entity"):
"entity"<|>"entity_name"<|>"entity_type"<|>"entity_description"

Relationship record format (MUST start with literal "relationship"):
"relationship"<|>"source_entity"<|>"target_entity"<|>"relation_description"

-Example-
Input text: "Paris is the capital of France."
Guidance: ["Country", "City"]##Country <|> has_capital <|> City
Output:
"entity"<|>"Paris"<|>"City"<|>"The capital city"##"entity"<|>"France"<|>"Country"<|>"A country in Europe"##"relationship"<|>"France"<|>"Paris"<|>"has capital"

-Real Data-
Guidance:
{Graph_pattern}

Input text:
{input_text}

Output:'''

# --- Auxiliary: extract only explicit visual/image-related phrases from the question (<|Answer|> wrapped) ---
IMAGE_FEATURE_PROMPT = r'''-Goal-
You are an AI assistant tasked with extracting explicit image descriptions and visual characteristics from given questions or tasks.

-Steps-
1. Carefully read and analyze the provided question or task.
2. Identify any explicit descriptions of images, visual elements, or distinctive visual characteristics that could be used to identify elements in an image.
3. Return a list of strings, where each string is an extracted image description or visual characteristic. If no relevant descriptions are present, return an empty list.
4. Wrap the final output list with <|Answer|> and <|\Answer|> tags.
5. The list conforms to Python syntax, using double quotation marks to wrap each item.

Important:
1. Extract descriptions that are clearly referring to visual elements, images, or distinctive visual characteristics of people or objects, include physical descriptions that could be used to identify someone or something in an image.
2. Do not infer or generate new descriptions; use only what is explicitly stated in the text, include the full phrase or sentence that describes the visual element.
3. Do not modify or paraphrase the extracted descriptions.
4. If multiple relevant descriptions are present, extract each one separately.
5. Remove question words (what, which, how, etc.) and their associated terms (color, size, etc.) from the extracted descriptions, only keep the concrete visual elements.
6. Do not include any explanations or commentary in your output, only the list of extracted descriptions wrapped in <|Answer|> and <|\Answer|> tags.

######################
-Examples-
######################
Example 1:

Input: What is the main architectural style of the Eiffel Tower?
################
Output: <|Answer|>[]<|\Answer|>
######################
Example 2:

Input: Which film did Emma Stone appear in first: the one where she's a woman wearing a red cape and flying over a city skyline, or the one where she has long wavy hair standing next to a man in a tuxedo? Also, is she the one with short blonde hair holding a bouquet of flowers in another movie?
################
Output: <|Answer|>["a woman wearing a red cape and flying over a city skyline",
"long wavy hair standing next to a man in a tuxedo",
"short blonde hair holding a bouquet of flowers"]<|\Answer|>
######################
Example 3:

Input: What is the nickname of the organization that was once a member of the East Coast Athletic Conference and has a tiger in its logo?
################
Output: <|Answer|>["A tiger in its logo."]<|\Answer|>
######################
Example 4:

Input:  Which of Jay's albums has a little girl holding a doll on the cover?
################
Output:<|Answer|>["a little girl holding a doll on the cover"]<|\Answer|>
######################
-Real Data-
######################
Input: {question}
######################
Output: {Your final output must be a list wrapped in <|Answer|> and <|\Answer|> tags, without any other explanation}'''
# --- Auxiliary: extract head entities for graph lookup (explicit mentions only) ---
HEAD_NODE_PROMPT = r'''-Goal-
Entity Extraction from Questions: Given a question, identify and extract the relevant entities that would be used to search a knowledge graph to answer the question.

-Steps-
1. Carefully read the provided question.
2. Identify key entities mentioned in the question that might be used to search a knowledge graph for information to answer the question.
3. Extract only the entities that are explicitly mentioned. Do not infer or assume any entities that are not directly stated.
4. Place the extracted entities within square brackets, separated by commas.
5. If no relevant entities are found, return empty square brackets [].
6. Do not include any explanations or commentary in your output, only the list of extracted descriptions wrapped in <|Answer|> and <|\Answer|> tags.
7. The list conforms to Python syntax, using double quotation marks to wrap each item.

Important: Only extract entities that are explicitly stated in the question. Do not infer, assume, or create any entities that are not directly mentioned, even if they seem logical or likely. When in doubt, err on the side of not extracting rather than inferring.

Output Format
<|Answer|>["entity1", "entity2", ...]<|\Answer|>

######################
-Examples-
######################
Example 1:

Input: What year was Elizabeth Matory the opponent of Charles Albert Ruppersberger?
Output: <|Answer|>["Elizabeth Matory", "Charles Albert Ruppersberger"]<|\Answer|>
######################
Example 2:

Input: Which system has a lower number for Japan of the virtual console systems: Game Boy Advance or the Japan-only console MSX?
Output: <|Answer|>["Game Boy Advance", "Japan-only console MSX"]<|\Answer|>
######################
Example 3:

Input: Who is a Japanese video game designer and producer, currently serving as the co-Representative Director of Nintendo, who gave production input to a video game for the Nintendo 64, originally released in 1996 along with the debut of the console
Output: <|Answer|>["Nintendo 64"]<|\Answer|>
######################
-Real Data-
######################
Input: {question}
######################
Output: {Your final output must be a list wrapped in <|Answer|> and <|\Answer|> tags, without any other explanation}'''
# --- Auxiliary: distill final short answer from (question, model draft answer) ---
GET_FINAL_ANSWER_PROMPT = r'''--Goal--
Extract the final answer from the given questions and answers.

Steps
1. Analyze a given question and a passage of text to find the final answer from the text.
2. Analyze the question carefully, Identify the type of question, whether it is a multiple-choice or judgmental question or any other type of question.
3. My final answer may be wrong or right, you need to carefully analyze the given question and then find the corresponding information from my answer and then give your final answer.
4. Answer the question as concisely as possible:
   - For yes/no questions, respond with just "Yes" or "No"
   - For other questions, provide the shortest possible answer, preferably a single word or phrase
   - Do not use full sentences unless absolutely necessary
   - For questions asking about quantities, only numbers should be answered
   - Omit any explanations or additional context

######################
-Real Data-
######################
Input:
Question: {question}
My answer: {my_answer}
######################
Output:{Your answer}'''
# --- Phase 5 (inference.py, WebQA/general): graph_to_str summary + full answer sentence from question ---
LLM_ANSWER_PROMPT = r'''Goal
Answer the question based on the knowledge graph. Provide a complete, self-contained answer sentence.

Steps:
1. Read the question carefully and identify what is being asked.
2. Look at the graph nodes and relationships for relevant information.
3. Write a complete sentence that directly answers the question.

Important:
- For Yes/No questions: start with "Yes" or "No", then explain in a complete sentence.
- For "Which" or comparison questions, state the chosen option and the reason.
- For descriptive questions, state the answer with relevant details.
- Always respond in a complete sentence, not just a word or phrase.
- Restate key terms from the question so the answer is self-contained.

Examples:
Question: Are the tops of the caps of two mushrooms the same color?
Answer: No, the tops of the caps of the two mushrooms are not the same color as each other.

Question: Are there any buildings shorter than the flag pole?
Answer: No, there are not any buildings shorter than the flag pole.

Question: Does the Cincinnati Music Hall have columns inside and outside?
Answer: Yes, the Cincinnati Music Hall has columns inside and outside.

Question: Which building has more windows, A or B?
Answer: Building A has more windows on the facing with its entrance.

Question: What is the capital of France?
Answer: The capital of France is Paris.

######################
-Real Data-
######################
Input:
Question: {question}
Knowledge Graph:
{GraphML}
######################
Output:'''
# --- Phase 5 (inference.py, MMQA): graph facts only; short spans/list-style answers for list-F1 ---
MMQA_ANSWER_PROMPT = r'''Goal
Answer the question using ONLY the facts that appear in the knowledge graph below.

Formatting (aligned with MMQA list-F1 evaluation):
1. Prefer the SHORTEST factual span: single word or short phrase unless the Gold requires more.
   Examples (good): Mask | The Bad News Bears | 1976 | José Terrón Peñaranda | Yes | No
2. Do NOT write full explanatory sentences unless the Gold is itself a sentence.
3. Do NOT restate or paraphrase the question.
4. For multiple disjoint answers:
   Return a comma-separated list, OR a compact JSON-like list ["a","b"].
   No sentences before or after the list.

If unknown from the graph, reply exactly with: unknown

######################
-Real Data-
######################
Input:
Question: {question}
Knowledge Graph:
{GraphML}
######################
Output:'''
# --- Auxiliary: markdown table + question → candidate entities (must appear in table) ---
TABLE_QA_PROMPT = r'''Goal
Given a question and a table in markdown format (some questions may require multiple entity jumps to get the answer), identify candidate entities from the description field of the knowledge graph table node that may help answer the question. And return the answer in the specified format.

Step:
1. analyze the question and infer the information needed to answer the question.
2. analyze the given Markdown-formatted table.
3. extract entities from the table that are relevant to the question, even though they do not answer the question.

Important:
1. The names of the extracted entities must appear in the content of the table.
2. Don't ignore any entities that are relevant to the question, even if they don't answer it.
3. Returns the name of the entity associated with the question, or the name of the entity associated with some information in the question
4. If you believe that there are no task entities in the table that are relevant to the question, return an empty list, but think about it carefully.
5. Return to list only
6. Wrap the final output in <|Answer|> and <|\Answer|> tags.
7. The list conforms to Python syntax, using double quotation marks to wrap each item.

Output Format:
<|Answer|>[List of entity names extracted from descriptions, or [] if none]<|\Answer|>

######################
-Examples-
######################
Example 1

Input:
Question: Where has Liu Xiang run in the 13.50s in the Games？
Table name: Liu Xiang International competition record
Table content(markdown):
| Year | Competition              | Position      | Event         | Notes                             |
|------|--------------------------|---------------|---------------|-----------------------------------|
| 2000 | World Junior Championships | 4th           | 110 m hurdles | 13.87 (wind: -0.1 m/s)            |
| 2001 | World University Games   | 1st           | 110 m hurdles | 13.33 seconds                     |
| 2001 | World Championships      | 4th (semis)   | 110 m hurdles | 13.51                            |
| 2001 | Chinese National Games   | 1st           | 110 m hurdles | 13.36                            |
| 2001 | East Asian Games         | 1st           | 110 m hurdles | 13.42 seconds                    |
#############
Output:
<|Answer|>["World University Games","Chinese National Games","East Asian Games"]<|\Answer|>
######################
-Real Data-
######################
Question: {Question}
Table name: {Table name}
Table content(markdown):
{Table content}
######################
Output:{Your final output must be a list wrapped in <|Answer|> and <|\Answer|> tags, without any other explanation}'''
# --- Auxiliary: decide if images are needed for the answer; emit specific queries per image entity ---
IMAGEQ_PROMPT = '''-Goal-
Image Query Assistant: Your task is to determine whether image information is needed to answer a given query. If it is needed, the relevant image modal entity must be identified and a specific question about the image must be asked to that image entity. Results are returned in a format that does not contain any additional information. Only return the list, don't give any other explanation.

Steps:
1. Read the Query: Carefully analyze the question to identify any implicit or explicit need for image information.
2. Analyze the Knowledge Graph: Review entities and relationships in the knowledge graph to determine their relevance to the query.
3. Determine Image Necessity: Decide if image information would help answer the question.
4. Formulate Image Queries: If image information is required, list the node names(d0) for this image modality and specific questions related to those images.

Output Format:
<|Answer|>[{entity_name:question}]<|\Answer|>
If there are no questions to ask, return an empty list, for example:
<|Answer|>[]<|\Answer|>

Important:
1. The entity_name MUST be the name of the entity in the image modal and match the actual name of the entity shown in the d0 field, and MUST NOT contain any identifiers or additional type information. The question shall be specific and directly related to the image associated with the entity.
2. Multiple queries should be included in the list if necessary.
3. If no image information is required to answer the query, return an empty list.
4. Return the results in a format that does not contain any extra information, entity_name that return nodes of other types are considered inexcusable.
######################
-Examples-
######################
Example 1:

Input:
Question: What is the main architectural style of the Eiffel Tower?
Knowledge Graph:
======= BEGIN: TEXT NODES BLOCK =======
Name: Eiffel Tower
Type: Structure
Description: Iconic wrought-iron lattice tower on the Champ de Mars in Paris, France. Completed in 1889, it stands 324 meters (1,063 ft) tall and is the most-visited paid monument in the world.
---
Name: Paris
Type: City
Description: Capital city of France, known for its art, culture, and historical landmarks including the Eiffel Tower.
---
======= END: TEXT NODES BLOCK =======

======= BEGIN: IMAGE NODES BLOCK =======
Name: Eiffel Tower
Type: image
Description: Photograph showing the full view of the Eiffel Tower, clearly displaying its intricate ironwork and overall structure.
---
======= END: IMAGE NODES BLOCK =======

======= BEGIN: TABLE NODES BLOCK =======
======= END: TABLE NODES BLOCK =======

======= BEGIN: RELATIONSHIPS BLOCK =======
Node 1 Name: Eiffel Tower
Node 1 Type: Structure
Node 2 Name: Paris
Node 2 Type: City
Relationship between Node 1 and Node 2: Located in
----------
======= END: RELATIONSHIPS BLOCK =======
#############
Output:
<|Answer|>[{"Eiffel Tower": "What architectural style or design features are prominently visible in this image of the Eiffel Tower?"}]<|\Answer|>
#############################
Example 2:

Input:
Question: What color is the Eiffel Tower?
Knowledge Graph:
======= BEGIN: TEXT NODES BLOCK =======
Name: Eiffel Tower
Type: Structure
Description: Iconic wrought-iron lattice tower on the Champ de Mars in Paris, France. Completed in 1889, it stands 324 meters (1,063 ft) tall and is the most-visited paid monument in the world.
---
Name: Paris
Type: City
Description: Capital city of France, known for its art, culture, and historical landmarks including the Eiffel Tower.
---
======= END: TEXT NODES BLOCK =======

======= BEGIN: IMAGE NODES BLOCK =======
Name: Eiffel Tower
Type: image
Description: Photograph showing the full view of the Eiffel Tower, clearly displaying its intricate ironwork and overall structure.
---
======= END: IMAGE NODES BLOCK =======

======= BEGIN: TABLE NODES BLOCK =======
======= END: TABLE NODES BLOCK =======

======= BEGIN: RELATIONSHIPS BLOCK =======
Node 1 Name: Eiffel Tower
Node 1 Type: Structure
Node 2 Name: Paris
Node 2 Type: City
Relationship between Node 1 and Node 2: Located in
----------
======= END: RELATIONSHIPS BLOCK =======
#############
Output:
<|Answer|>[{"Eiffel Tower": "What color is the Eiffel Tower in this image? Please provide a description."}]<|\Answer|>
######################
-Real Data-
######################
Input:
Question: {question}
Knowledge Graph:
{GraphML}
######################
Output:{Your final output must be a list wrapped in <|Answer|> and <|\Answer|> tags, without any other explanation}'''
# --- Auxiliary: very concise factual answer from image + caption context ---
IMAGE_QA_PROMPT = r'''-Goal-
Provide extremely concise and factual answers about the content of the image and the image captions

-Steps
1. First describe the image in the context of the image name and image content.
1. If the answer to the question can be found in the picture, answer the question.
2. if the answer to the question cannot be found in the picture, briefly  state that the picture does not contain the content of the question
3. use declarative sentences, do not offer additional information or invite further questions.

Important:
1. Use both the image content and the provided image Image_title as factual information.
2. Image_title is the name of that image and is perfectly correct
######################
-Real Data-
######################
Input:
Question: {question}
Image_title: {title}
######################
Output:'''
# --- Auxiliary: whether the question is answerable from the current graph alone (Yes/No) and why ---
DETERMINE_ANSWER_PROMPT = r'''Goal
Given a question and a knowledge graph, determine if the question can be answered based on the current knowledge graph. Pay close attention to the specific details requested in the question.

Steps
1. Analyze the Question: Thoroughly examine the provided question to understand the information needs.
2. Carefully analyze the given knowledge map and determine if you can answer the question based on the information in the map.
3. If the question can be answered based on the knowledge graph and give the correct answer in an explanation that is as concise as possible.
4. If you can't answer, or can't answer very accurately, please return to No with an explanation,by think step by step.

Important:
You should carefully read the constraints in the question and each node in the given knowledge graph. You should list the constraints and find the corresponding information step by step before answering.

Output Format
Reason: {Reason for inability to answer Or final answer, think step by step}
Answer: [Yes/No]
######################
-Real Data-
######################
Input:
Knowledge Graph:
{GraphML}
######################
Question: {question}
Output:{Strictly formatted returns}'''
# --- Auxiliary: pick one of three candidate images that best matches the description (1/2/3) ---
SELECT_IMAGE_PROMPT = r'''--Objective--
You are given three pictures (labeled 1, 2 and 3) and a description in English. Analyze the content of all three pictures carefully. It is possible that multiple pictures fit the description well. You will need to look closely at the given description in English and determine which one fits the given description the best.

Steps:
1. Look closely at the details of all three pictures.
2. Compare the content of each picture with the given description.
3. Choose the picture that best matches the description among all three options.
4. After "Answer:", write only "1", "2" or "3" without any additional text.
5. Explain your choice succinctly in the "Reason:" section.
6. Do not include any comments or information other than these two required lines.
7. Remember to keep your answers concise and strictly adhere to the required two-line format.
8. Your answer should be consistent with your reasoning.

Output Format:
Reason: [explain your choice]
Answer: [1, 2 or 3]

Description: {description}
Output:{Returns in the format}'''
