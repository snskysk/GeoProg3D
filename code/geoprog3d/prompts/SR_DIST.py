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
"""Question: <Building A> is within 70 meters of <Building B>.
Program:
SEG_TO0=GetLandmarkSeg(query='Building A')
SEG_FROM0=GetLandmarkSeg(query='Building B')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} < 70 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: <Building A> is more than 50 meters away from <Building B>.
Program:
SEG_TO0=GetLandmarkSeg(query='Building A')
SEG_FROM0=GetLandmarkSeg(query='Building B')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 50 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: The parking lot closest to <Building A> is within 100 meters of <Building B>.
Program:
CLOSEST_FROM0=GetLandmarkSeg(query='Building A')
AREAS0=GetSimilarAreas(query='parking lot',area_filter_flag=False)
SEG0=GetCluster(seg=AREAS0,cluster_type='CLOSEST',from=CLOSEST_FROM0)
CLOSEST_TO0=SegAround(seg=SEG0)
AREAS1=GetSimilarAreas(query='parking lot',area_filter_flag=True,area=CLOSEST_TO0)
SEG_TO0=GetCluster(seg=AREAS1,cluster_type='BIGGEST')
SEG_FROM0=GetLandmarkSeg(query='Building B')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} < 100 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: The parking lot closest to <Building A> is within 150 meters of <Building A>.
Program:
CLOSEST_FROM0=GetLandmarkSeg(query='Building A')
AREAS0=GetSimilarAreas(query='parking lot',area_filter_flag=False)
SEG0=GetCluster(seg=AREAS0,cluster_type='CLOSEST',from=CLOSEST_FROM0)
CLOSEST_TO0=SegAround(seg=SEG0)
AREAS1=GetSimilarAreas(query='parking lot',area_filter_flag=True,area=CLOSEST_TO0)
SEG_TO0=GetCluster(seg=AREAS1,cluster_type='BIGGEST')
SEG_FROM0=GetLandmarkSeg(query='Building A')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} < 150 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: The parking lot closest to <Building A> is within 150 meters of yellow tower.
Program:
CLOSEST_FROM0=GetLandmarkSeg(query='Building A')
AREAS0=GetSimilarAreas(query='parking lot',area_filter_flag=False)
SEG0=GetCluster(seg=AREAS0,cluster_type='CLOSEST',from=CLOSEST_FROM0)
CLOSEST_TO0=SegAround(seg=SEG0)
AREAS1=GetSimilarAreas(query='parking lot',area_filter_flag=True,area=CLOSEST_TO0)
SEG_TO0=GetCluster(seg=AREAS1,cluster_type='BIGGEST')
AREAS2=GetSimilarAreas(query='yellow tower',area_filter_flag=False)
SEG_FROM0=GetCluster(seg=AREAS2,cluster_type='BIGGEST')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} < 150 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: The park to the northeast of <Building A> is within 120 meters of yellow tower.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='northeast')
AREAS0=GetSimilarAreas(query='park',area_filter_flag=True,area=SEG1)
SEG_TO0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
AREAS1=GetSimilarAreas(query='yellow tower',area_filter_flag=False)
SEG_FROM0=GetCluster(seg=AREAS1,cluster_type='BIGGEST')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} < 120 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: The park northeast of <Building A> is within 200 meters of clock tower.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='northeast')
AREAS0=GetSimilarAreas(query='park',area_filter_flag=True,area=SEG1)
SEG_TO0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
AREAS1=GetSimilarAreas(query='clock tower',area_filter_flag=False)
SEG_FROM0=GetCluster(seg=AREAS1,cluster_type='BIGGEST')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} < 200 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: The park directly east of <Building A> is within 200 meters of clock tower.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='directly east')
AREAS0=GetSimilarAreas(query='park',area_filter_flag=True,area=SEG1)
SEG_TO0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
AREAS1=GetSimilarAreas(query='clock tower',area_filter_flag=False)
SEG_FROM0=GetCluster(seg=AREAS1,cluster_type='BIGGEST')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} < 200 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: The parking lot between <Building A> and <Building B> is within 100 meters of <Building B>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=GetLandmarkSeg(query='Building B')
SEG2=SegBetween(seg_first=SEG0,seg_second=SEG1)
AREAS0=GetSimilarAreas(query='parking lot',area_filter_flag=True,area=SEG2)
SEG_TO0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
SEG_FROM0=GetLandmarkSeg(query='Building A')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} < 100 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: <Building C> is closer to <Building A> than <Building B>
Program:
SEG_TO0=GetLandmarkSeg(query='Building C')
SEG_FROM0=GetLandmarkSeg(query='Building A')
SEG_TO1=GetLandmarkSeg(query='Building B')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
ANSWER1=MeasureDist(from=SEG1, to=SEG_TO1)
ANSWER2=EVAL(expr="'yes' if {ANSWER0} < {ANSWER1} else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: <Building A> is closer to yellow tower than <Building B>
Program:
SEG_TO0=GetLandmarkSeg(query='Building A')
AREAS0=GetSimilarAreas(query='yellow tower',area_filter_flag=False)
SEG_FROM0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
SEG_TO1=GetLandmarkSeg(query='Building B')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
ANSWER1=MeasureDist(from=SEG_FROM0, to=SEG_TO1)
ANSWER2=EVAL(expr="'yes' if {ANSWER0} < {ANSWER1} else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: <Building A> is closer to big stadium than <Building B>
Program:
SEG_TO0=GetLandmarkSeg(query='Building A')
AREAS0=GetSimilarAreas(query='big stadium',area_filter_flag=False)
SEG_FROM0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
SEG_TO1=GetLandmarkSeg(query='Building B')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
ANSWER1=MeasureDist(from=SEG_FROM0, to=SEG_TO1)
ANSWER2=EVAL(expr="'yes' if {ANSWER0} < {ANSWER1} else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: <Building A> is closer to <Building B> than clock tower
Program:
SEG_TO0=GetLandmarkSeg(query='Building A')
SEG_FROM0=GetLandmarkSeg(query='Building B')
AREAS0=GetSimilarAreas(query='clock tower',area_filter_flag=False)
SEG_TO1=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
ANSWER1=MeasureDist(from=SEG_FROM0, to=SEG_TO1)
ANSWER2=EVAL(expr="'yes' if {ANSWER0} < {ANSWER1} else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: sports stadium is closer to <Building B> than <Building A>
Program:
AREAS0=GetSimilarAreas(query='clock tower',area_filter_flag=False)
SEG_TO0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
SEG_FROM0=GetLandmarkSeg(query='Building B')
SEG_TO1=GetLandmarkSeg(query='Building A')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
ANSWER1=MeasureDist(from=SEG_FROM0, to=SEG_TO1)
ANSWER2=EVAL(expr="'yes' if {ANSWER0} < {ANSWER1} else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: The parking lot to the north of <Building A> is closer to <Building B> than the parking lot to the south of <Building A>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='north')
AREAS0=GetSimilarAreas(query='parking lot',area_filter_flag=True,area=SEG1)
SEG_TO0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
SEG_FROM0=GetLandmarkSeg(query='Building B')
SEG2=GetLandmarkSeg(query='Building A')
SEG3=SegDirection(seg=SEG2,query='south')
AREAS1=GetSimilarAreas(query='parking lot',area_filter_flag=True,area=SEG3)
SEG_TO1=GetCluster(seg=AREAS1,cluster_type='BIGGEST')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
ANSWER1=MeasureDist(from=SEG_FROM0, to=SEG_TO1)
ANSWER2=EVAL(expr="'yes' if {ANSWER0} < {ANSWER1} else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: The parking lot to the north of <Building A> is closer to <Building A> than the parking lot to the south of <Building A>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='north')
AREAS0=GetSimilarAreas(query='parking lot',area_filter_flag=True,area=SEG1)
SEG_TO0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
SEG_FROM0=GetLandmarkSeg(query='Building A')
SEG2=GetLandmarkSeg(query='Building A')
SEG3=SegDirection(seg=SEG2,query='south')
AREAS1=GetSimilarAreas(query='parking lot',area_filter_flag=True,area=SEG3)
SEG_TO1=GetCluster(seg=AREAS1,cluster_type='BIGGEST')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
ANSWER1=MeasureDist(from=SEG_FROM0, to=SEG_TO1)
ANSWER2=EVAL(expr="'yes' if {ANSWER0} < {ANSWER1} else 'no'")
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