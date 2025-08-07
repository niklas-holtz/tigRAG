
from tigrag import TemporalInfluenceGraph, TigParam

if __name__ == '__main__':
    params = TigParam(
        embedding_model_name="local",
        llm_name="llama"
    )

    tig = TemporalInfluenceGraph(
        working_dir="./data",
        query_param=params
    )

    #tig.insert('Das ist ein langer Test!')

    test_text = """
    Life in a big city can be both exciting and overwhelming. With endless entertainment options, diverse cultures, and constant movement, there is always something happening. People from all walks of life come together, creating a unique blend of traditions, languages, and lifestyles.

However, city life also comes with its challenges. The fast pace can lead to stress and burnout, and the constant noise and crowds may leave little room for peace and quiet. Many people struggle to find balance between work, social life, and personal time in such a demanding environment.

Despite the difficulties, many choose to stay because of the opportunities a big city offers. From career growth to educational institutions and networking possibilities, the benefits can outweigh the drawbacks for those who thrive in a dynamic setting.
"""
    tig.insert(test_text)