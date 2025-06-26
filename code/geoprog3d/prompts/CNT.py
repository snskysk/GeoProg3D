PROMPT = """Think step by step to carry out the instruction.

The names of buildings and streets are enclosed in <>.

Instruction: How many skyscrapers are there?
Program:
ANSWER0=ObjCounting(query='skyscrapers',area_filter_flag=False)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How many skyscrapers are there directly east of <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='directly east')
ANSWER0=ObjCounting(query='skyscrapers',area_filter_flag=True,area=SEG1)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How many skyscrapers, directly east of <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='directly east')
ANSWER0=ObjCounting(query='skyscrapers',area_filter_flag=True,area=SEG1)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How many skyscrapers are there to the northwest of <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='northwest')
ANSWER0=ObjCounting(query='skyscrapers',area_filter_flag=True,area=SEG1)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How many skyscrapers, to the northwest of <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='northwest')
ANSWER0=ObjCounting(query='skyscrapers',area_filter_flag=True,area=SEG1)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How many skyscrapers, around <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0)
SIMPLE_ANSWER0=ObjCounting(query='skyscrapers',area_filter_flag=True,area=SEG1)
FINAL_RESULT=RESULT(var=SIMPLE_ANSWER0)

Instruction: How many skyscrapers are there within 250 meters of <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0,size=250)
ANSWER0=ObjCounting(query='skyscrapers',area_filter_flag=True,area=SEG1)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How many skyscrapers, within 250 meters of <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0,size=250)
SIMPLE_ANSWER0=ObjCounting(query='skyscrapers',area_filter_flag=True,area=SEG1)
FINAL_RESULT=RESULT(var=SIMPLE_ANSWER0)

Instruction: How many cars are there in the parking spaces around <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0)
AREAS0=GetSimilarAreas(query='parking spaces',area_filter_flag=True,area=SEG1)
SEG2=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
ANSWER0=ObjCounting(query='cars',area_filter_flag=True,area=SEG2)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How many cars are in the parking space closest to <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
AREAS0=GetSimilarAreas(query='parking space',area_filter_flag=False)
SEG1=GetCluster(seg=AREAS0,cluster_type='CLOSEST',from=SEG0)
SEG2=SegAround(seg=SEG1)
AREAS1=GetSimilarAreas(query='parking space',area_filter_flag=True,area=SEG2)
ANSWER0=ObjCounting(query='cars',area_filter_flag=True,area=AREAS1)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How many cars, in the parking space closest to <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
AREAS0=GetSimilarAreas(query='parking space',area_filter_flag=False)
SEG1=GetCluster(seg=AREAS0,cluster_type='CLOSEST',from=SEG0)
SEG2=SegAround(seg=SEG1)
AREAS1=GetSimilarAreas(query='parking space',area_filter_flag=True,area=SEG2)
SIMPLE_ANSWER0=ObjCounting(query='cars',area_filter_flag=True,area=AREAS1)
FINAL_RESULT=RESULT(var=SIMPLE_ANSWER0)

Instruction: How many cars are there on the road closest to <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
AREAS0=GetSimilarAreas(query='road',area_filter_flag=False)
SEG1=GetCluster(seg=AREAS0,cluster_type='CLOSEST',from=SEG0)
SEG2=SegAround(seg=SEG1)
AREAS1=GetSimilarAreas(query='road',area_filter_flag=True,area=SEG2)
ANSWER0=ObjCounting(query='cars',area_filter_flag=True,area=AREAS1)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How many cars, on the road closest to <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
AREAS0=GetSimilarAreas(query='road',area_filter_flag=False)
SEG1=GetCluster(seg=AREAS0,cluster_type='CLOSEST',from=SEG0)
SEG2=SegAround(seg=SEG1)
AREAS1=GetSimilarAreas(query='road',area_filter_flag=True,area=SEG2)
SIMPLE_ANSWER0=ObjCounting(query='cars',area_filter_flag=True,area=AREAS1)
FINAL_RESULT=RESULT(var=SIMPLE_ANSWER0)

Instruction: How many bridges are there between <Building A> and <Building B>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=GetLandmarkSeg(query='Building B')
SEG2=SegBetween(seg_first=SEG0,seg_second=SEG1)
ANSWER0=ObjCounting(query='bridges',area_filter_flag=True,area=SEG2)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How many bridges over the river, between <Building A> and <Building B>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=GetLandmarkSeg(query='Building B')
SEG2=SegBetween(seg_first=SEG0,seg_second=SEG1)
SENTENCE_ANSWER0=ObjCounting(query='bridges over the river',area_filter_flag=True,area=SEG2)
FINAL_RESULT=RESULT(var=SENTENCE_ANSWER0)

Instruction: How many bridges over the river, around <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0)
SENTENCE_ANSWER0=ObjCounting(query='bridges over the river',area_filter_flag=True,area=SEG1)
FINAL_RESULT=RESULT(var=SENTENCE_ANSWER0)

Instruction: How many cars for advertising, around <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0)
SENTENCE_ANSWER0=ObjCounting(query='cars for advertising',area_filter_flag=True,area=SEG1)
FINAL_RESULT=RESULT(var=SENTENCE_ANSWER0)

Instruction: How many cars for advertising, to the directly west of <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='directly west')
SENTENCE_ANSWER0=ObjCounting(query='cars for advertising',area_filter_flag=True,area=SEG1)
FINAL_RESULT=RESULT(var=SENTENCE_ANSWER0)

Instruction: How many bridges across the lake, east of <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='east')
SENTENCE_ANSWER0=ObjCounting(query='bridges across the lake',area_filter_flag=True,area=SEG1)
FINAL_RESULT=RESULT(var=SENTENCE_ANSWER0)

Instruction: How many cars are in the parking space that is closest to the tennis courts within 100m of <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0,size=100)
AREAS0=GetSimilarAreas(query='tennis courts',area_filter_flag=True,area=SEG1)
SEG2=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
AREAS1=GetSimilarAreas(query='parking space',area_filter_flag=False)
SEG3=GetCluster(seg=AREAS1,cluster_type='CLOSEST',from=SEG2)
SEG4=SegAround(seg=SEG3)
AREAS2=GetSimilarAreas(query='parking space',area_filter_flag=True,area=SEG4)
ANSWER0=ObjCounting(query='cars',area_filter_flag=True,area=AREAS2)
FINAL_RESULT=RESULT(var=ANSWER0)


Instruction: {instruction}
Program:

"""

