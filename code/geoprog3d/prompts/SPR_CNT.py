import random

GQA_CURATED_EXAMPLES=[
"""Question: There is a bridge between <Building A> and <Building B>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=GetLandmarkSeg(query='Building B')
SEG2=SegBetween(seg_first=SEG0,seg_second=SEG1)
ANSWER0=ObjCounting(query='bridge',area_filter_flag=True,area=SEG2)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There are more than 5 buildings.
Program:
ANSWER0=ObjCounting(query='building',area_filter_flag=False)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 5 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There are more than five buildings.
Program:
ANSWER0=ObjCounting(query='building',area_filter_flag=False)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 5 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There are five or more buildings.
Program:
ANSWER0=ObjCounting(query='building',area_filter_flag=False)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} >= 5 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There are at least two cars.
Program:
ANSWER0=ObjCounting(query='car',area_filter_flag=False)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} >= 2 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There are less than 5 buildings.
Program:
ANSWER0=ObjCounting(query='building',area_filter_flag=False)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} < 5 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There are less than five buildings.
Program:
ANSWER0=ObjCounting(query='building',area_filter_flag=False)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} < 5 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There are more than 10 cars around <Building A>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0)
ANSWER0=ObjCounting(query='car',area_filter_flag=True,area=SEG1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 10 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There are more than 3 cars near <Building A>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0)
ANSWER0=ObjCounting(query='car',area_filter_flag=True,area=SEG1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 3 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There are more than three cars near <Building A>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0)
ANSWER0=ObjCounting(query='car',area_filter_flag=True,area=SEG1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 3 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There are 3 or more cars near <Building A>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0)
ANSWER0=ObjCounting(query='car',area_filter_flag=True,area=SEG1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} >= 3 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There are three or more cars near <Building A>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0)
ANSWER0=ObjCounting(query='car',area_filter_flag=True,area=SEG1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} >= 3 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There are less than 3 cars near <Building A>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0)
ANSWER0=ObjCounting(query='car',area_filter_flag=True,area=SEG1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} < 3 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There are less than three cars near <Building A>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0)
ANSWER0=ObjCounting(query='car',area_filter_flag=True,area=SEG1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} < 3 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There are 3 or less cars near <Building A>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0)
ANSWER0=ObjCounting(query='car',area_filter_flag=True,area=SEG1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} <= 3 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There are three or less cars near <Building A>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0)
ANSWER0=ObjCounting(query='car',area_filter_flag=True,area=SEG1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} <= 3 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There are more than five buildings to the south of <Building A>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='south')
ANSWER0=ObjCounting(query='building',area_filter_flag=True,area=SEG1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 5 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There are more than three cars west of <Building A>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='west')
ANSWER0=ObjCounting(query='car',area_filter_flag=True,area=SEG1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 3 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There are more than 10 cars between the soccer court and <Building A>.
Program:
AREAS0=GetSimilarAreas(query='soccer court',area_filter_flag=False)
SEG0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
SEG1=GetLandmarkSeg(query='Building A')
SEG2=SegBetween(seg_first=SEG0,seg_second=SEG1)
ANSWER0=ObjCounting(query='car',area_filter_flag=True,area=SEG2)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 10 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There are more than 20 cars between the soccer court closest to the yellow skyscraper and <Building A>.
Program:
AREAS0=GetSimilarAreas(query='yellow skyscraper',area_filter_flag=False)
SEG0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
AREAS1=GetSimilarAreas(query='soccer court',area_filter_flag=False)
SEG1=GetCluster(seg=AREAS1,cluster_type='CLOSEST',from=SEG0)
SEG2=SegAround(seg=SEG1)
AREAS2=GetSimilarAreas(query='soccer court',area_filter_flag=True,area=SEG2)
SEG3=GetLandmarkSeg(query='Building A')
SEG4=SegBetween(seg_first=AREAS2,seg_second=SEG3)
ANSWER0=ObjCounting(query='car',area_filter_flag=True,area=SEG4)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 20 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There are more than 30 cars around the soccer field to the directly south of <Building A>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='directly south')
AREAS0=GetSimilarAreas(query='soccer court',area_filter_flag=True,area=SEG1)
SEG2=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
SEG3=SegAround(seg=SEG2)
ANSWER0=ObjCounting(query='car',area_filter_flag=True,area=SEG1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 30 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: The parking lot closest to <Building A> is within 100 meters of <Building B>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
AREAS0=GetSimilarAreas(query='parking lot',area_filter_flag=False)
SEG1=GetCluster(seg=AREAS0,cluster_type='CLOSEST',from=SEG0)
SEG2=SegAround(seg=SEG1)
AREAS1=GetSimilarAreas(query='parking lot',area_filter_flag=True,area=SEG2)
SEG3=GetLandmarkSeg(query='Building B')
ANSWER0=MeasureDist(from=SEG3, to=AREAS1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} < 100 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
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