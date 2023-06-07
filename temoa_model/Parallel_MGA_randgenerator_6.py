from __future__ import division
import time
from collections import defaultdict
from joblib import Parallel, delayed
import multiprocessing
import sys
import os
from shutil import copyfile
import sqlite3
from pyomo.environ import *
from pyomo.core import Objective, Var, Constraint
from pyomo.opt import SolverFactory
from IPython import embed as IP
from io import StringIO
# cwd = os.getcwd()
# os.chdir('temoa_model/')
from temoa_model import temoa_create_model
from temoa_mga_Hadi import SlackedObjective_rule, ActivityObj_rule
from temoa_rules import ActivityByTech_Constraint
# os.chdir(cwd)
import argparse
import datetime
import ast

def MGA_parallel( Perfect_foresight_OF, dat, output, s, c, write_to_db):
	
	print('Creating model instance..')
	M = temoa_create_model()
	M.del_component('TotalCost')
	M.V_ActivityByTech = Var(M.tech_all, domain=NonNegativeReals)
	M.ActivityByTechConstraint = Constraint(M.tech_all, rule=ActivityByTech_Constraint)	
	data1 = DataPortal(model = M)
	data1.load(filename=dat)

	start_time_instance = time.time()
	MGAinstance = M.create_instance(data1)
	
	MGAinstance.PreviousSlackedObjective = Constraint(
	rule=None,
	expr=SlackedObjective_rule( MGAinstance, Perfect_foresight_OF , s ),
	noruleinit=True
	)

	MGAinstance.SecondObj = Objective(
	expr=ActivityObj_rule( MGAinstance, 'randgenerator', {} ),
	noruleinit=True,
	sense=minimize)
	
	print('Solving..')
	optimizer = SolverFactory('gurobi')

	# These are the settings for perfect foresight TS
	optimizer.options['threads'] = 4
	optimizer.options['method'] = 2 # could try simplex - method 1
	optimizer.options['crossover'] = 0
	# optimizer.options['BarConvTol'] = 1.e-5
	# optimizer.options['FeasibilityTol'] = 1.e-6
	# optimizer.options['PreDual'] = 0
	# optimizer.options['NumericFocus'] = 2
	optimizer.options['BarHomogeneous'] = 1

	print('time taken to set up instance and optimizer', time.time() - start_time_instance)
	start_time_solve = time.time()
	results = optimizer.solve(MGAinstance, tee = True)
	print('time taken to solve is', time.time() - start_time_solve)

	MGAinstance.solutions.load_from(results)

	print (MGAinstance.PreviousSlackedObjective.slack(),'---------------------', s, c)
	start_time_write = time.time()
	write_to_db (MGAinstance, output, s, c)
	print('time taken to wrtie results is', time.time() - start_time_write)



def Base_MGA_run (dat, output, write_to_db, slack, ii):

	M = temoa_create_model()
	data1 = DataPortal(model = M)
	data1.load(filename=dat)
	
	print('Creating base MGA instance..')
	start_time_instance = time.time()
	Baseinstance = M.create_instance(data1)

	optimizer = SolverFactory('gurobi')

	# These are the settings for perfect foresight TS
	optimizer.options['threads'] = 4
	optimizer.options['method'] = 2 # could try simplex - method 1
	optimizer.options['crossover'] = 0
	# optimizer.options['BarConvTol'] = 1.e-5
	# optimizer.options['FeasibilityTol'] = 1.e-6
	# optimizer.options['PreDual'] = 0
	# optimizer.options['NumericFocus'] = 2
	optimizer.options['BarHomogeneous'] = 1

	print('time taken to set up instance and optimizer', time.time() - start_time_instance)
	
	start_time_solve = time.time()
	print('Solving base instance..')
	results = optimizer.solve(Baseinstance, tee = True)
	print('time taken to solve is', time.time() - start_time_solve)
	
	OF = value (Baseinstance.TotalCost)
	start_time_write = time.time()
	write_to_db(Baseinstance, output, slack, ii)
	print('time taken to wrtie results is', time.time() - start_time_write)
	
	return OF

def write_to_db (MGAinstance, output, slack, c):
	print('Reading solved values..')
	svars = defaultdict( lambda: defaultdict( float ))   
	con_info = list()
	epsilon = 1e-4
	emission_keys = { (r, i, t, v, o) : set() for r, e, i, t, v, o in MGAinstance.EmissionActivity }
	for r, e, i, t, v, o in MGAinstance.EmissionActivity:
		emission_keys[(r, i, t, v, o)].add(e)
	P_0 = min( MGAinstance.time_optimize )
	P_e = MGAinstance.time_future.last()
	GDR = value( MGAinstance.GlobalDiscountRate )
	# MLL = MGAinstance.ModelLoanLife
	MPL = MGAinstance.ModelProcessLife
	LLN = MGAinstance.LifetimeLoanProcess
	x   = 1 + GDR
	# vflow_in is defined only for storage techs
	for r, p, s, d, i, t, v, o in MGAinstance.V_FlowIn:
		val_in = value( MGAinstance.V_FlowIn[r, p, s, d, i, t, v, o] )
		# if abs(val_in) < epsilon: continue
		if abs(val_in) > epsilon:
			svars['V_FlowIn'][r, p, s, d, i, t, v, o] = val_in


	for r, p, s, d, i, t, v, o in MGAinstance.V_FlowOut:
		val_out = value( MGAinstance.V_FlowOut[r, p, s, d, i, t, v, o] )
		# if abs(val_out) < epsilon: continue
		if abs(val_out) > epsilon:
			svars['V_FlowOut'][r, p, s, d, i, t, v, o] = val_out

		if t not in MGAinstance.tech_storage:
			val_in = value( MGAinstance.V_FlowOut[r, p, s, d, i, t, v, o] ) / value(MGAinstance.Efficiency[r, i, t, v, o]) 
			svars['V_FlowIn'][r, p, s, d, i, t, v, o] = val_in

		if (r, i, t, v, o) not in emission_keys: 
			continue

		emissions = emission_keys[r, i, t, v, o]
		for e in emissions:
			evalue = val_out * MGAinstance.EmissionActivity[r, e, i, t, v, o]
			svars[ 'V_EmissionActivityByPeriodAndProcess' ][r, p, e, t, v] += evalue
	
	for r, p, i, t, v, o in MGAinstance.V_FlowOutAnnual:
		for s in MGAinstance.time_season:
			for d in MGAinstance.time_of_day:
				val_out = value( MGAinstance.V_FlowOutAnnual[r, p, i, t, v, o] ) * value( MGAinstance.SegFrac[s , d ])
				# if abs(val_out) < epsilon: continue
				if abs(val_out) > epsilon:
					svars['V_FlowOut'][r, p, s, d, i, t, v, o] = val_out
					svars['V_FlowIn'][r, p, s, d, i, t, v, o] = val_out / value(MGAinstance.Efficiency[r, i, t, v, o])
				if (r, i, t, v, o) not in emission_keys: 
					continue
				emissions = emission_keys[r, i, t, v, o]
				for e in emissions:
					evalue = val_out * MGAinstance.EmissionActivity[r, e, i, t, v, o]
					svars[ 'V_EmissionActivityByPeriodAndProcess' ][r, p, e, t, v] += evalue	
	
	for r, p, s, d, i, t, v, o in MGAinstance.V_Curtailment:		
		val = value( MGAinstance.V_Curtailment[r, p, s, d, i, t, v, o] )
		# if abs(val) < epsilon: continue
		if abs(val) > epsilon:
			svars['V_Curtailment'][r, p, s, d, i, t, v, o] = val
			svars['V_FlowIn'][r, p, s, d, i, t, v, o] = (val + value( MGAinstance.V_FlowOut[r, p, s, d, i, t, v, o] )) / value(MGAinstance.Efficiency[r, i, t, v, o])

	# Extract optimal decision variable values related to capacity:
	for r, t, v in MGAinstance.V_Capacity:
		val = value( MGAinstance.V_Capacity[r, t, v] )
		# if abs(val) < epsilon: continue
		if abs(val) > epsilon:
			svars['V_Capacity'][r, t, v] = val

	for r, p, t in MGAinstance.V_CapacityAvailableByPeriodAndTech:
		val = value( MGAinstance.V_CapacityAvailableByPeriodAndTech[r, p, t] )
		# if abs(val) < epsilon: continue
		if abs(val) > epsilon:
			svars['V_CapacityAvailableByPeriodAndTech'][r, p, t] = val

	# Calculate model costs:	
	# This is a generic workaround.  Not sure how else to automatically discover 
    # the objective name
	objs = list(MGAinstance.component_data_objects( Objective ))
	obj_name, obj_value = objs[0].getname(True), value( objs[0] )	
	svars[ 'Objective' ]["('"+obj_name+"')"] = obj_value

	for r, t, v in MGAinstance.CostInvest.sparse_iterkeys():   # Returns only non-zero values
	
		icost = value( MGAinstance.V_Capacity[r, t, v] )
		# if abs(icost) < epsilon: continue
		if abs(icost) > epsilon:
			icost *= value( MGAinstance.CostInvest[r, t, v] )*(
				(
					1 -  x**( -min( value(MGAinstance.LifetimeProcess[r, t, v]), P_e - v ) )
				)/(
					1 -  x**( -value( MGAinstance.LifetimeProcess[r, t, v] ) ) 
				)
			)
			svars[	'Costs'	][ r, 'V_UndiscountedInvestmentByProcess', t, v] += icost

			icost *= value( MGAinstance.LoanAnnualize[r, t, v] )
			icost *= (
			  value( LLN[r, t, v] ) if not GDR else
			    (x **(P_0 - v + 1) * (1 - x **(-value( LLN[r, t, v] ))) / GDR)
			)

			svars[	'Costs'	][ r, 'V_DiscountedInvestmentByProcess', t, v] += icost

	for r, p, t, v in MGAinstance.CostFixed.sparse_iterkeys():
		fcost = value( MGAinstance.V_Capacity[r, t, v] )
		# if abs(fcost) < epsilon: continue
		if abs(fcost) > epsilon:
			fcost *= value( MGAinstance.CostFixed[r, p, t, v] )
			svars[	'Costs'	][ r, 'V_UndiscountedFixedCostsByProcess', t, v] += fcost
			
			fcost *= (
			  value( MPL[r, p, t, v] ) if not GDR else
			    (x **(P_0 - p + 1) * (1 - x **(-value( MPL[r, p, t, v] ))) / GDR)
			)

			svars[	'Costs'	][ r, 'V_DiscountedFixedCostsByProcess', t, v] += fcost
		
	for r, p, t, v in MGAinstance.CostVariable.sparse_iterkeys():
		if t not in MGAinstance.tech_annual:
			vcost = sum(
				value (MGAinstance.V_FlowOut[r, p, S_s, S_d, S_i, t, v, S_o])
				for S_i in MGAinstance.processInputs[r, p, t, v]
				for S_o in MGAinstance.ProcessOutputsByInput[r, p, t, v, S_i]
				for S_s in MGAinstance.time_season
				for S_d in MGAinstance.time_of_day
			)
		
		else:
			vcost = sum(
				value (MGAinstance.V_FlowOutAnnual[r, p, S_i, t, v, S_o])
				for S_i in MGAinstance.processInputs[r, p, t, v]
				for S_o in MGAinstance.ProcessOutputsByInput[r, p, t, v, S_i]
			)			
		# if abs(vcost) < epsilon: continue
		if abs(vcost) > epsilon:
			vcost *= value( MGAinstance.CostVariable[r, p, t, v] )
			svars[	'Costs'	][ r, 'V_UndiscountedVariableCostsByProcess', t, v] += vcost

			vcost *= (
			  value( MPL[r, p, t, v] ) if not GDR else
			    (x **(P_0 - p + 1) * (1 - x **(-value( MPL[r, p, t, v] ))) / GDR)
			  )
			svars[	'Costs'	][ r, 'V_DiscountedVariableCostsByProcess', t, v] += vcost

	var_list = []
	for vgroup, values in sorted( svars.items() ):
		for vindex, val in sorted( values.items() ):
			if isinstance( vindex, tuple ):
				vindex = ','.join( str(i) for i in vindex )
			var_list.append(( '{}[{}]'.format(vgroup, vindex), val ))
	ostream = StringIO()
	for i, (v, val) in enumerate( var_list ):
			ipart, fpart = repr(float(val)).split('.')
			var_list[i] = (ipart, fpart, v)
	cell_lengths = ( map(len, l[:-1] ) for l in var_list )
	max_lengths = map(max, zip(*cell_lengths))   # max length of each column
	fmt = u'  {{:>{:d}}}.{{:<{:d}}}  {{}}\n'.format( *max_lengths )
	
	for row in var_list:
		ostream.write( fmt.format(*row) )
	
	
	tables = { "V_FlowIn"   : "Output_VFlow_In",  \
			   "V_FlowOut"  : "Output_VFlow_Out", \
			   "V_Curtailment"  : "Output_Curtailment", \
			   "V_CapacityAvailableByPeriodAndTech"   : "Output_CapacityByPeriodAndTech",  \
			   "V_EmissionActivityByPeriodAndProcess" : "Output_Emissions", \
			   "Objective"  : "Output_Objective", \
			   "Costs"      : "Output_Costs" }
	
	
	dot_loc = output.find('.')


	if len(str(slack)[2:]) == 2:
		string = '_slack'+str(slack)[2:]+'0'
		if c < 0:
			scenario = str(slack)[2:]+'0'
		else:
			scenario = str(slack)[2:]+'0'+"_mga_"+str(c)
	elif len(str(slack)[2:]) == 1:
		string = '_slack'+str(slack)[2:]+'00'
		if c < 0:
			scenario = str(slack)[2:]+'00'
		else:
			scenario = str(slack)[2:]+'00'+"_mga_"+str(c)
	else:
		string = '_slack'+str(slack)[2:]
		if c < 0:
			scenario = str(slack)[2:]
		else:
			scenario = str(slack)[2:]+"_mga_"+str(c)


	if c < 0: #Base MGA case
		MGA_db = output[:dot_loc]+string+output[dot_loc:]

	else:
		MGA_db = output[:dot_loc]+string+'_'+str(c)+output[dot_loc:]
	copyfile(output, MGA_db)

	con = sqlite3.connect(MGA_db)
	cur = con.cursor()   # A database cursor enables traversal over DB records
	con.text_factory = str # This ensures data is explored with UTF-8 encoding
	
	for table in svars.keys() :
		if table in tables :
			cur.execute("SELECT DISTINCT scenario FROM '"+tables[table]+"'")

			if table == 'Objective' : # Only table without sector info
				for key in svars[table].keys():
					key_str = str(key) # only 1 row to write
					key_str = key_str[1:-1] # Remove parentheses
					cur.execute("INSERT INTO "+tables[table]+" \
								VALUES('"+scenario+"',"+key_str+", \
								"+str(svars[table][key])+");")
			else : # First add 'NULL' for sector then update
				for key in svars[table].keys() : # Need to loop over keys (rows)
					key_str = str(key)
					key_str = key_str[1:-1] # Remove parentheses						
					if table != 'Costs':
						cur.execute("INSERT INTO "+tables[table]+ \
									" VALUES('"+str(key[0])+"', '"+scenario+"','NULL', \
										"+key_str[key_str.find(',')+1:]+","+str(svars[table][key])+");")	
					else:						
						key_str = str((key[0],key[2],key[3]))
						key_str = key_str[1:-1] # Remove parentheses
						cur.execute("INSERT INTO "+tables[table]+ \
									" VALUES('"+str(key[1])+"', '"+scenario+"','NULL', \
									"+key_str+","+str(svars[table][key])+");")																																	
				cur.execute("UPDATE "+tables[table]+" SET sector = \
							(SELECT technologies.sector FROM technologies \
							WHERE "+tables[table]+".tech = technologies.tech);")

	# for table in svars.keys() :
	# 	if table in tables :
	# 		cur.execute("SELECT DISTINCT scenario FROM '"+tables[table]+"'")
	# 		if table == 'Objective' : # Only table without sector info
	# 			for key in svars[table].keys():
	# 				key_str = str(key) # only 1 row to write
	# 				key_str = key_str[1:-1] # Remove parentheses
	# 				cur.execute("INSERT INTO "+tables[table]+" \
	# 							VALUES('"+scenario+"',"+key_str+", \
	# 							"+str(svars[table][key])+");")
	# 		else : # First add 'NULL' for sector then update
	# 			for key in svars[table].keys() : # Need to loop over keys (rows)
	# 				key_list = list(key)
	# 				_region_save = key_list.pop(0) # Need to pop out region index -> order of values supplied to SQL insert statements are subject to unique constraints
	# 				key_list_str = str(key_list)
	# 				key_list_str = key_list_str[1:-1] # Remove square brackets of lists
	# 				_region_save_str = str(_region_save)
	# 				cur.execute("INSERT INTO "+tables[table]+ \
	# 							" VALUES('" +_region_save_str+"','" +scenario+"','NULL', \
	# 							"+key_list_str+","+str(svars[table][key])+");")
	# 			cur.execute("UPDATE "+tables[table]+" SET sector = \
	# 						(SELECT technologies.sector FROM technologies \
	# 						WHERE "+tables[table]+".tech = technologies.tech);")
	con.isolation_level = None # Need isolation level workaround for Python 3.6 onwards. See: https://github.com/ghaering/pysqlite/issues/109

	con.execute("VACUUM")
	con.isolation_level = ''			
	con.commit()
	con.close()


def temp_function(k):
	# print(datetime.datetime.now())
	# print("this was the second run")
	time.sleep(100)

def pipeline(args):

	input_file = args.input_file
	output_file = args.output_file
	num_iter = args.num_iter
	identifier = args.identifier 
	identifier = ast.literal_eval(identifier) # for Adi: lists are interepreted as strings from parse args - can try setting agg_argument to type list in argparser

	slack = 0.05
	num_cores = multiprocessing.cpu_count() # for Adi: need to talk to Lucas about n_jobs=-1 implementation and performance improvements relative to node scaling

	con = sqlite3.connect(output_file)
	cur = con.cursor()
	tables_list = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
	tables_list = [i[0] for i in tables_list]
	key_tables = ['Output_VFlow_In','Output_VFlow_Out','Output_CapacityByPeriodAndTech','Output_Emissions','Output_Objective','Output_Curtailment','Output_Costs','technologies']
	for table in tables_list:
		if table not in key_tables:
			cur.execute("DROP TABLE "+table+";")

	# cur.execute("VACUUM")			
	con.commit()
	con.close()

	# Parallel(n_jobs=1)(delayed(Base_MGA_run)(input_file, output_file, write_to_db, slack, ii) for ii in identifier)
	# Base_MGA_OF = 66685017.99
	# Base_MGA_OF = 39402466.57
	Base_MGA_OF = 37218653.67
	# Base_MGA_OF = Base_MGA_run(input_file, output_file, write_to_db, slack, identifier[0])
	Parallel(n_jobs=1)(delayed(MGA_parallel)(Base_MGA_OF, input_file, output_file, slack, ii, write_to_db) for ii in identifier)

	# start_time = time.time()
	# Morris_Objectives = Parallel(n_jobs=10)(delayed(temp_function)(i) for i in range(0,num_iter))
	# print("time it took to run this is ", time.time() - start_time)


def parse_args(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Accept input .dat and ( empty ) output .sqlite3 files for running MGA"
    )
    parser.add_argument(
        "-i", "--input_file",
        dest="input_file",
        required=True,
        default=None,
        help="Specify the prepared .dat file"
    )
    parser.add_argument(
        "-o", "--output_file",
        dest="output_file",
        required=True,
        help="Specify the output .sqlite file"
    )
    parser.add_argument(
        "-n", "--num_iter",
        dest="num_iter",
        default=1,
        type=int,
        help="How many iterations of MGA are to be run? (default = 1)"
    )
    # parser.add_argument(
    #     "-c", "--identifier",
    #     dest="identifier",
    #     nargs = "*",
    #     default=None,
    #     help="Identifier code for MGA iterations (specification level: non essential)"
    # )
    parser.add_argument(
        "-c", "--identifier",
        dest="identifier",
        default=None,
        help="Identifier code for MGA iterations (specification level: non essential)"
    )      
    if arg_list:
        return parser.parse_args(arg_list)
    else:
        return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    pipeline(args)



