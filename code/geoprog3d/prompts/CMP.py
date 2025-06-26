import random

GQA_CURATED_EXAMPLES=[
"""Question: Which is taller, <Building A> or <Building B>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
ANSWER0=GetAreaHeight(seg=SEG0)
SEG1=GetLandmarkSeg(query='Building B')
ANSWER1=GetAreaHeight(seg=SEG1)
ANSWER2=EVAL(expr="'Building A' if {ANSWER0} > {ANSWER1} else 'Building B'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Which is taller, the yellow tower or <Building A>?
Program:
AREAS0=GetSimilarAreas(query='yellow tower',area_filter_flag=False)
SEG0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
ANSWER0=GetAreaHeight(seg=SEG0)
SEG1=GetLandmarkSeg(query='Building A')
ANSWER1=GetAreaHeight(seg=SEG1)
ANSWER2=EVAL(expr="'the yellow tower' if {ANSWER0} > {ANSWER1} else 'Building A'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Which is taller, the skyscraper around the yellow tower or <Building A>?
Program:
AREAS0=GetSimilarAreas(query='yellow tower',area_filter_flag=False)
SEG0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
SEG1=SegAround(seg=SEG0)
AREAS1=GetSimilarAreas(query='skyscraper',area_filter_flag=True,area=SEG1)
SEG2=GetCluster(seg=AREAS1,cluster_type='BIGGEST')
ANSWER0=GetAreaHeight(seg=SEG2)
SEG3=GetLandmarkSeg(query='Building A')
ANSWER1=GetAreaHeight(seg=SEG3)
ANSWER2=EVAL(expr="'the skyscraper around the yellow tower' if {ANSWER0} > {ANSWER1} else 'Building A'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Which is taller, the building to the west of the yellow tower or <Building A>?
Program:
AREAS0=GetSimilarAreas(query='yellow tower',area_filter_flag=False)
SEG0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
SEG1=SegDirection(seg=SEG0,query='west')
AREAS1=GetSimilarAreas(query='building',area_filter_flag=True,area=SEG1)
SEG2=GetCluster(seg=AREAS1,cluster_type='BIGGEST')
ANSWER0=GetAreaHeight(seg=SEG2)
SEG3=GetLandmarkSeg(query='Building A')
ANSWER1=GetAreaHeight(seg=SEG3)
ANSWER2=EVAL(expr="'the building to the west of the yellow tower' if {ANSWER0} > {ANSWER1} else 'Building A'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Which is taller, the building to the directly north of <Building A> or <Building B>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='directly north')
AREAS0=GetSimilarAreas(query='building',area_filter_flag=True,area=SEG1)
SEG2=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
ANSWER0=GetAreaHeight(seg=SEG2)
SEG3=GetLandmarkSeg(query='Building B')
ANSWER1=GetAreaHeight(seg=SEG3)
ANSWER2=EVAL(expr="'the building to the directly north of <Building A>' if {ANSWER0} > {ANSWER1} else 'Building B'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Which is taller, the house that is closest to <Building A>, or <Building B>?
Program:
CLOSEST_FROM0=GetLandmarkSeg(query='Building A')
AREAS0=GetSimilarAreas(query='house',area_filter_flag=False)
SEG0=GetCluster(seg=AREAS0,cluster_type='CLOSEST',from=CLOSEST_FROM0)
CLOSEST_TO0=SegAround(seg=SEG0)
AREAS1=GetSimilarAreas(query='house',area_filter_flag=True,area=CLOSEST_TO0)
ANSWER0=GetAreaHeight(seg=AREAS1)
SEG2=GetLandmarkSeg(query='Building B')
ANSWER1=GetAreaHeight(seg=SEG2)
ANSWER2=EVAL(expr="'the house that is closest to <Building A>' if {ANSWER0} > {ANSWER1} else 'Building B'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Which is taller, the building that is closest to <Building A>, or <Building B>?
Program:
CLOSEST_FROM0=GetLandmarkSeg(query='Building A')
AREAS0=GetSimilarAreas(query='building',area_filter_flag=False)
SEG0=GetCluster(seg=AREAS0,cluster_type='CLOSEST',from=CLOSEST_FROM0)
CLOSEST_TO0=SegAround(seg=SEG0)
AREAS1=GetSimilarAreas(query='building',area_filter_flag=True,area=CLOSEST_TO0)
ANSWER0=GetAreaHeight(seg=AREAS1)
SEG1=GetLandmarkSeg(query='Building B')
ANSWER1=GetAreaHeight(seg=SEG1)
ANSWER2=EVAL(expr="'the building that is closest to <Building A>' if {ANSWER0} > {ANSWER1} else 'Building B'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Which is taller, the building that is closest to the yellow tower, or <Building A>?
Program:
AREAS0=GetSimilarAreas(query='yellow tower',area_filter_flag=False)
CLOSEST_FROM0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
AREAS1=GetSimilarAreas(query='building',area_filter_flag=False)
SEG0=GetCluster(seg=AREAS1,cluster_type='CLOSEST',from=CLOSEST_FROM0)
CLOSEST_TO0=SegAround(seg=SEG0)
AREAS2=GetSimilarAreas(query='building',area_filter_flag=True,area=CLOSEST_TO0)
ANSWER0=GetAreaHeight(seg=AREAS2)
SEG1=GetLandmarkSeg(query='Building A')
ANSWER1=GetAreaHeight(seg=SEG1)
ANSWER2=EVAL(expr="'the building that is closest to the yellow tower' if {ANSWER0} > {ANSWER1} else 'Building A'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Which is taller, the building to the directly north of <Building A>, or the house that is closest to <Building B>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='directly north')
AREAS0=GetSimilarAreas(query='building',area_filter_flag=True,area=SEG1)
SEG2=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
ANSWER0=GetAreaHeight(seg=SEG2)
CLOSEST_FROM0=GetLandmarkSeg(query='Building B')
AREAS1=GetSimilarAreas(query='house',area_filter_flag=False)
SEG3=GetCluster(seg=AREAS1,cluster_type='CLOSEST',from=CLOSEST_FROM0)
CLOSEST_TO0=SegAround(seg=SEG3)
AREAS2=GetSimilarAreas(query='house',area_filter_flag=True,area=CLOSEST_TO0)
ANSWER1=GetAreaHeight(seg=AREAS2)
ANSWER2=EVAL(expr="'the building to the directly north of <Building A>' if {ANSWER0} > {ANSWER1} else 'the house that is closest to <Building B>'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Which is taller, the house between <Building A> and <Building B> or <Building C>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=GetLandmarkSeg(query='Building B')
SEG2=SegBetween(seg_first=SEG0,seg_second=SEG1)
AREAS0=GetSimilarAreas(query='house',area_filter_flag=True,area=SEG2)
SEG3=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
ANSWER0=GetAreaHeight(seg=SEG3)
SEG4=GetLandmarkSeg(query='Building C')
ANSWER1=GetAreaHeight(seg=SEG4)
ANSWER2=EVAL(expr="'the house between <Building A> and <Building B>' if {ANSWER0} > {ANSWER1} else 'Building C'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Which is taller, the tallest object around <Building A> or <Building B>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0)
ANSWER0=GetAreaHeight(seg=SEG1)
SEG2=GetLandmarkSeg(query='Building B')
ANSWER1=GetAreaHeight(seg=SEG2)
ANSWER2=EVAL(expr="'the tallest object around <Building A>' if {ANSWER0} > {ANSWER1} else 'Building B'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Which is taller, the tallest object within 100 meters of <Building A> or <Building B>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0,size=100)
ANSWER0=GetAreaHeight(seg=SEG1)
SEG2=GetLandmarkSeg(query='Building B')
ANSWER1=GetAreaHeight(seg=SEG2)
ANSWER2=EVAL(expr="'the tallest object around <Building A>' if {ANSWER0} > {ANSWER1} else 'Building B'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Which is taller, the tallest object within 60 meters of <Building A> or <Building B>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0,size=60)
ANSWER0=GetAreaHeight(seg=SEG1)
SEG2=GetLandmarkSeg(query='Building B')
ANSWER1=GetAreaHeight(seg=SEG2)
ANSWER2=EVAL(expr="'the tallest object around <Building A>' if {ANSWER0} > {ANSWER1} else 'Building B'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Which is taller, the tallest object to the west of <Building A> or <Building B>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='west')
ANSWER0=GetAreaHeight(seg=SEG1)
SEG2=GetLandmarkSeg(query='Building B')
ANSWER1=GetAreaHeight(seg=SEG2)
ANSWER2=EVAL(expr="'the tallest object around <Building A>' if {ANSWER0} > {ANSWER1} else 'Building B'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Which is taller, <Building A> or the tallest object to the southeast of <Building B>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
ANSWER0=GetAreaHeight(seg=SEG0)
SEG1=GetLandmarkSeg(query='Building B')
SEG2=SegDirection(seg=SEG1,query='southeast')
ANSWER1=GetAreaHeight(seg=SEG2)
ANSWER2=EVAL(expr="'Building A' if {ANSWER0} > {ANSWER1} else 'the tallest object to the southeast of <Building B>'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Which is taller, the tallest object directly north of <Building A> or the tallest object directly south of <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='directly north')
ANSWER0=GetAreaHeight(seg=SEG1)
SEG2=GetLandmarkSeg(query='Building A')
SEG3=SegDirection(seg=SEG2,query='directly south')
ANSWER1=GetAreaHeight(seg=SEG3)
ANSWER2=EVAL(expr="'the tallest object directly north of <Building A>' if {ANSWER0} > {ANSWER1} else 'the tallest object directly south of <Building A>'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
]

def create_prompt(inputs,num_prompts=8,method='random',seed=42,group=0):
    if method=='all':
        prompt_examples = GQA_CURATED_EXAMPLES
    elif method=='random':
        random.seed(seed)
        prompt_examples = random.sample(GQA_CURATED_EXAMPLES,num_prompts)
    else:
        raise NotImplementedError

    prompt_examples = '\n'.join(prompt_examples)
    prompt_examples = f'Think step by step to answer the question. The names of buildings and streets are enclosed in <>.\n\n{prompt_examples}'


    return prompt_examples + "\nQuestion: {question}\nProgram:".format(**inputs)