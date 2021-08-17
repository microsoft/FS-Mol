""" Queries and fieldnames used to extract all data from ChEMBL """

CHEMBL_ASSAY_PROTEIN = (
    "SELECT s.canonical_smiles as smiles, act.pchembl_value as pchembl,"
    " act.standard_value as standard_value,"
    " act.standard_units as standard_units,"
    " act.standard_relation as standard_relation,"
    " act.activity_comment as activity_comment,"
    " a.chembl_id as chembl_id,"
    " a.assay_type as assay_type,"
    " a.assay_organism as organism,"
    " a.confidence_score as confidence_score,"
    " td.tid as target_id,"
    " td.pref_name as target,"
    " tt.target_type as target_type,"
    " protcls.protein_class_id as protein_id,"
    " protcls.pref_name as protein_class_name,"
    " protcls.short_name as protein_short_name,"
    " protcls.class_level as protein_class_level,"
    " protcls.protein_class_desc as protein_class_desc"
    " FROM assays a"
    " JOIN activities act ON a.assay_id = act.assay_id"
    " JOIN compound_structures s ON act.molregno = s.molregno"
    " JOIN target_dictionary td on td.tid = a.tid"
    " JOIN target_components tc on td.tid = tc.tid"
    " JOIN target_type tt on tt.target_type = td.target_type"
    " JOIN component_class compcls on tc.component_id = compcls.component_id"
    " JOIN protein_classification protcls on protcls.protein_class_id = compcls.protein_class_id"
    " AND a.chembl_id = {}"
)

DISTINCT_TABLES = {
    "activity_comment": ("SELECT DISTINCT d.chembl_id, d.activity_comment FROM ({}) as d;"),
    "standard_unit": ("SELECT DISTINCT d.chembl_id, d.standard_units FROM ({}) as d;"),
    "target_id": ("SELECT DISTINCT  d.chembl_id, d.target_id FROM ({}) as d;"),
    "protein_class_level": (
        " SELECT DISTINCT d.chembl_id, d.protein_class_level as protein_class_level"
        " FROM ({}) as d;"
    ),
    "target_type": (
        " SELECT DISTINCT d.chembl_id, d.target_type as target_type" " FROM ({}) as d;"
    ),
}

EXTENDED_SINGLE_ASSAY_NOPROTEIN = (
    "SELECT s.canonical_smiles as smiles,"
    " act.pchembl_value as pchembl,"
    " act.standard_value as value,"
    " act.standard_units as units,"
    " act.standard_relation as relation,"
    " act.activity_comment as comment,"
    " a.chembl_id as chembl_id,"
    " a.assay_type as assay_type,"
    " a.assay_organism as organism,"
    " a.confidence_score as confidence_score,"
    " a.assay_cell_type as cell_type,"
    " a.assay_tissue as tissue"
    " FROM assays a"
    " JOIN activities act on a.assay_id = act.assay_id"
    " JOIN compound_structures s"
    " ON act.molregno = s.molregno AND a.chembl_id = {}"
)

COUNT_QUERIES = {
    "num_activity_comment": "SELECT count(e.activity_comment) as num_activity_comment FROM ({}) as e GROUP BY e.chembl_id;",
    "num_standard_unit": "SELECT count(e.standard_units) as num_standard_unit FROM ({}) as e GROUP BY e.chembl_id;",
    "num_target_id": "SELECT count(e.target_id) as num_target_id FROM ({}) as e GROUP BY e.chembl_id;",
    "num_protein_class_level": "SELECT count(e.protein_class_level) as num_protein_class_level FROM ({}) as e GROUP BY e.chembl_id;",
    "num_target_type": "SELECT count(e.target_type) as num_target_type FROM ({}) as e GROUP BY e.chembl_id;",
}

FIELDNAMES = [
    "smiles",
    "pchembl",
    "standard_value",
    "standard_units",
    "standard_relation",
    "activity_comment",
    "chembl_id",
    "assay_type",
    "assay_organism",
    "confidence_score",
]

PROTEIN_FIELDS = [
    "target_id",
    "target",
    "target_type",
    "protein_id",
    "protein_class_name",
    "protein_short_name",
    "protein_class_level",
    "protein_class_desc",
]

CELL_FIELDS = [
    "assay_cell_type",
    "assay_tissue",
]

SUMMARY_FIELDNAMES = [
    "activity_comment",
    "standard_unit",
    "target_id",
    "protein_class_level",
    "target_type",
]
COUNTED_SUMMARY_FIELDNAMES = [
    "chembl_id",
    "num_activity_comment",
    "num_standard_unit",
    "num_target_id",
    "num_protein_class_level",
    "num_target_type",
    "size",
]
