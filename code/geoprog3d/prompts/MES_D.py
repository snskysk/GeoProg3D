PROMPT = """Think step by step to carry out the instruction.

The names of buildings and streets are enclosed in <>.

Instruction: How far is <Building A> from <Building B>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=GetLandmarkSeg(query='Building B')
ANSWER0=MeasureDist(from=SEG1, to=SEG0)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How far is the green car near the red letter sign from <Building A>?
Program:
AREAS0=GetSimilarAreas(query='red letter sign',area_filter_flag=False)
SEG0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
SEG1=SegAround(seg=SEG0)
AREAS1=GetSimilarAreas(query='green car',area_filter_flag=True,area=SEG1)
SEG_TO0=GetCluster(seg=AREAS1, cluster_type='BIGGEST')
SEG_FROM0=GetLandmarkSeg(query='Building A')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How far away is the parking lot closest to <Building A> from <Building B>?
Program:
CLOSEST_FROM0=GetLandmarkSeg(query='Building A')
AREAS0=GetSimilarAreas(query='parking lot',area_filter_flag=False)
SEG0=GetCluster(seg=AREAS0,cluster_type='CLOSEST',from=CLOSEST_FROM0)
CLOSEST_TO0=SegAround(seg=SEG0)
AREAS1=GetSimilarAreas(query='parking lot',area_filter_flag=True,area=CLOSEST_TO0)
SEG_TO0=GetCluster(seg=AREAS1,cluster_type='BIGGEST')
SEG_FROM0=GetLandmarkSeg(query='Building B')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How far away is the parking lot closest to <Building A> from <Building A>?
Program:
CLOSEST_FROM0=GetLandmarkSeg(query='Building A')
AREAS0=GetSimilarAreas(query='parking lot',area_filter_flag=False)
SEG0=GetCluster(seg=AREAS0,cluster_type='CLOSEST',from=CLOSEST_FROM0)
CLOSEST_TO0=SegAround(seg=SEG0)
AREAS1=GetSimilarAreas(query='parking lot',area_filter_flag=True,area=CLOSEST_TO0)
SEG_TO0=GetCluster(seg=AREAS1,cluster_type='BIGGEST')
SEG_FROM0=GetLandmarkSeg(query='Building A')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How far is the skyscraper within 30 meters around the red letter sign from <Building A>?
Program:
AREAS0=GetSimilarAreas(query='red letter sign',area_filter_flag=False)
SEG0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
SEG1=SegAround(seg=SEG0,size=30)
AREAS1=GetSimilarAreas(query='skyscraper',area_filter_flag=True,area=SEG1)
SEG_TO0=GetCluster(seg=AREAS1, cluster_type='BIGGEST')
SEG_FROM0=GetLandmarkSeg(query='Building A')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How far away is the building within 100 meters around the yellow tower from <Building A>?
Program:
AREAS0=GetSimilarAreas(query='yellow tower',area_filter_flag=False)
SEG0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
SEG1=SegAround(seg=SEG0,size=100)
AREAS1=GetSimilarAreas(query='building',area_filter_flag=True,area=SEG1)
SEG_TO0=GetCluster(seg=AREAS1, cluster_type='BIGGEST')
SEG_FROM0=GetLandmarkSeg(query='Building A')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How far away is the telephone box closest to <Building B> from <Building B>?
Program:
CLOSEST_FROM0=GetLandmarkSeg(query='Building B')
AREAS0=GetSimilarAreas(query='telephone box',area_filter_flag=False)
SEG0=GetCluster(seg=AREAS0,cluster_type='CLOSEST',from=CLOSEST_FROM0)
CLOSEST_TO0=SegAround(seg=SEG0)
AREAS1=GetSimilarAreas(query='telephone box',area_filter_flag=True,area=CLOSEST_TO0)
SEG_TO0=GetCluster(seg=AREAS1,cluster_type='BIGGEST')
SEG_FROM0=GetLandmarkSeg(query='Building B')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How far is the park to the east of <Building A> from <Building B>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='east')
AREAS0=GetSimilarAreas(query='park',area_filter_flag=True,area=SEG1)
SEG_TO0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
SEG_FROM0=GetLandmarkSeg(query='Building B')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How far is the park to the east of <Building A> from <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='east')
AREAS0=GetSimilarAreas(query='park',area_filter_flag=True,area=SEG1)
SEG_TO0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
SEG_FROM0=GetLandmarkSeg(query='Building A')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How far is the park to the east of <Building A> from the park to the west of <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='east')
AREAS0=GetSimilarAreas(query='park',area_filter_flag=True,area=SEG1)
SEG_TO0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
SEG2=GetLandmarkSeg(query='Building A')
SEG3=SegDirection(seg=SEG2,query='west')
AREAS1=GetSimilarAreas(query='park',area_filter_flag=True,area=SEG3)
SEG_FROM0=GetCluster(seg=AREAS1,cluster_type='BIGGEST')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How far is the park between <Building A> and <Building B> from <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=GetLandmarkSeg(query='Building B')
SEG2=SegBetween(seg_first=SEG0,seg_second=SEG1)
AREAS0=GetSimilarAreas(query='park',area_filter_flag=True,area=SEG2)
SEG_TO0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
SEG_FROM0=GetLandmarkSeg(query='Building A')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How far is the school between yellow tower and <Building B> from <Building A>?
Program:
AREAS0=GetSimilarAreas(query='yellow tower',area_filter_flag=False)
SEG0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
SEG1=GetLandmarkSeg(query='Building B')
SEG2=SegBetween(seg_first=SEG0,seg_second=SEG1)
AREAS1=GetSimilarAreas(query='school',area_filter_flag=True,area=SEG2)
SEG_TO0=GetCluster(seg=AREAS1,cluster_type='BIGGEST')
SEG_FROM0=GetLandmarkSeg(query='Building A')
ANSWER0=MeasureDist(from=SEG_FROM0, to=SEG_TO0)
FINAL_RESULT=RESULT(var=ANSWER0)



Instruction: {instruction}
Program:

"""