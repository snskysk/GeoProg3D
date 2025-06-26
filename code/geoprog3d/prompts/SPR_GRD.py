import random

GQA_CURATED_EXAMPLES=[
"""Question: There is a bridge between <Building A> and <Building B>.
Program:
SEG_FIRST0=GetLandmarkSeg(query='Building A')
SEG_SECOND0=GetLandmarkSeg(query='Building B')
SEG_BETWEEN0=SegBetween(seg_first=SEG_FIRST0,seg_second=SEG_SECOND0)
ANSWER0=ObjCounting(query='bridge',area_filter_flag=True,area=SEG_BETWEEN0)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
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
SEG_FIRST0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
SEG_SECOND0=GetLandmarkSeg(query='Building A')
SEG_BETWEEN0=SegBetween(seg_first=SEG_FIRST0,seg_second=SEG_SECOND0)
ANSWER0=ObjCounting(query='car',area_filter_flag=True,area=SEG_BETWEEN0)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 10 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There are more than 20 cars between the soccer court that is closest to the yellow skyscraper and <Building A>.
Program:
AREAS0=GetSimilarAreas(query='yellow skyscraper',area_filter_flag=False)
CLOSEST_FROM0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
AREAS1=GetSimilarAreas(query='soccer court',area_filter_flag=False)
CLOSEST_CLUSTER0=GetCluster(seg=AREAS1,cluster_type='CLOSEST',from=CLOSEST_FROM0)
CLOSEST_TO0=SegAround(seg=CLOSEST_CLUSTER0)
SEG_FIRST0=GetSimilarAreas(query='soccer court',area_filter_flag=True,area=CLOSEST_TO0)
SEG_SECOND0=GetLandmarkSeg(query='Building A')
SEG_BETWEEN0=SegBetween(seg_first=SEG_FIRST0,seg_second=SEG_SECOND0)
ANSWER0=ObjCounting(query='car',area_filter_flag=True,area=SEG_BETWEEN0)
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
"""Question: There is a telephone box.
Program:
ANSWER0=ObjCounting(query='telephone box',area_filter_flag=False)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There is a blue awning tent shop.
Program:
ANSWER0=ObjCounting(query='blue awning tent shop',area_filter_flag=False)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There is a telephone box around <Building A>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0)
ANSWER0=ObjCounting(query='telephone box',area_filter_flag=True,area=SEG1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There is a telephone box near <Building A>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0,size=100)
ANSWER0=ObjCounting(query='telephone box',area_filter_flag=True,area=SEG1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There is a building with red wall to the southwest of <Building A>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='southwest')
ANSWER0=ObjCounting(query='building with red wall',area_filter_flag=True,area=SEG1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There is a reddish-colored building southwest of <Building A>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='southwest')
ANSWER0=ObjCounting(query='reddish-colored building',area_filter_flag=True,area=SEG1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There is a reddish-colored building to the directly east of <Building A>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='directly east')
ANSWER0=ObjCounting(query='reddish-colored building',area_filter_flag=True,area=SEG1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There is a reddish-colored building to the directly east of <Building A>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='directly east')
ANSWER0=ObjCounting(query='reddish-colored building',area_filter_flag=True,area=SEG1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There is a bridge between <Building A> and reddish-colored building around <Building B>.
Program:
SEG_FIRST0=GetLandmarkSeg(query='Building A')
SEG0=GetLandmarkSeg(query='Building B')
SEG1=SegAround(seg=SEG0)
SEG_SECOND0=GetSimilarAreas(query='reddish-colored building',area_filter_flag=True,area=SEG1)
SEG_BETWEEN0=SegBetween(seg_first=SEG_FIRST0,seg_second=SEG_SECOND0)
ANSWER0=ObjCounting(query='bridge',area_filter_flag=True,area=SEG_BETWEEN0)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There is a black sedan-type automobile between building with yellow wall to the southeast of <Building A> and <Building B>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='southeast')
SEG_FIRST0=GetSimilarAreas(query='building with yellow wall',area_filter_flag=True,area=SEG1)
SEG_SECOND0=GetLandmarkSeg(query='Building B')
SEG_BETWEEN0=SegBetween(seg_first=SEG_FIRST0,seg_second=SEG_SECOND0)
ANSWER0=ObjCounting(query='black sedan-type automobile',area_filter_flag=True,area=SEG_BETWEEN0)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There is a bridge between <Building A> and a tall reddish brown building that is closest to <Building B>.
Program:
SEG_FIRST0=GetLandmarkSeg(query='Building A')
CLOSEST_FROM0=GetLandmarkSeg(query='Building B')
AREAS0=GetSimilarAreas(query='tall reddish brown building',area_filter_flag=False)
CLOSEST_CLUSTER0=GetCluster(seg=AREAS0,cluster_type='CLOSEST',from=CLOSEST_FROM0)
CLOSEST_TO0=SegAround(seg=CLOSEST_CLUSTER0)
SEG_SECOND0=GetSimilarAreas(query='tall reddish brown building',area_filter_flag=True,area=CLOSEST_TO0)
SEG_BETWEEN0=SegBetween(seg_first=SEG_FIRST0,seg_second=SEG_SECOND0)
ANSWER0=ObjCounting(query='bridge',area_filter_flag=True,area=SEG_BETWEEN0)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: There is a red wall on <Building A>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
ANSWER0=ObjCounting(query='red wall',area_filter_flag=True,area=SEG0)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
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