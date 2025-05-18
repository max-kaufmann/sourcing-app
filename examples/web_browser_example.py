def parse_tags(text: str, tag_name: str) -> Optional[str]:
    """Extract content between specified tags."""
    pattern = rf"<{tag_name}>\n?(.*?)\n?</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


# We tokenize the documents and add the index of the fact to the dataset
def train_set_to_hf_dict(doc: SynthDocument) -> dict[str, Any]:
    hf_dict = asdict(doc)
    hf_dict["prompt"] = ""
    hf_dict["completion"] = doc.text
    hf_dict["idx"] = doc.fact.idx
    hf_dict["fact"] = asdict(doc.fact)
    hf_dict["type"] = "atomic_fact"
    del hf_dict["text"]
    return hf_dict


def cache_dataset(dataset: Dataset) -> Dataset:
    cache_file = (
        Path(HF_DATASETS_CACHE)
        / "user"
        / "synthetic_pretraining_docs"
        / f"{dataset._fingerprint}"
    )  # type: ignore
    if not cache_file.exists():
        dataset.save_to_disk(cache_file)
    return load_from_disk(cache_file)  # type: ignore


def prep_eval_dataset(
    city: City,
    fact: Fact,
    few_shot_example_cities: list[City],
    num_few_shot_examples: int,
    random_generator: random.Random | None = None,
    second_hop_inferred_fact_template: tuple[
        str, str
    ] = SECOND_HOP_INFERRED_FACT_TEMPLATE,
) -> dict[str, Any]:
    few_shot_example_cities_for_this_fact = [
        c for c in few_shot_example_cities if c != city
    ]
    if random_generator is None:
        random_generator = random.Random(42)

    few_shot_example_cities_for_this_fact = random_generator.sample(
        few_shot_example_cities_for_this_fact, num_few_shot_examples
    )

    few_shot_examples = [
        (
            (
                second_hop_inferred_fact_template[0]
                + second_hop_inferred_fact_template[1]
            ).format(**asdict(city))
        )
        for city in few_shot_example_cities_for_this_fact
    ]

    prompt = (
        "\n".join(few_shot_examples)
        + "\n"
        + second_hop_inferred_fact_template[0].format(**asdict(city))
    )
    completion = second_hop_inferred_fact_template[1].format(**asdict(city))

    return {
        "prompt": prompt,
        "completion": completion,
        "city": asdict(city),
        "fact": asdict(fact),
        "idx": fact.idx,
    }
