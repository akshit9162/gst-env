"""GST rules encoded for the mock GSTR-3B environment.

Simplified on purpose; every simplification is listed in README.md and must be
reviewed by a practising CA before tasks are sold.
"""

# Post GST 2.0 slabs (effective 22 Sep 2025). 0 = nil-rated/exempt.
RATES = [0, 5, 18, 40]

# State codes used in GSTINs. The filer is always registered in HOME_STATE.
STATES = {
    "36": "Telangana",
    "37": "Andhra Pradesh",
    "29": "Karnataka",
    "27": "Maharashtra",
    "33": "Tamil Nadu",
    "07": "Delhi",
}
HOME_STATE = "36"

# Inward supplies whose ITC is blocked under section 17(5) (simplified subset).
BLOCKED_CATEGORIES = {
    "motor_vehicle": "Motor vehicle for personal transport",
    "food_catering": "Food and beverages / outdoor catering",
    "club_membership": "Club or health/fitness membership",
    "personal_use": "Goods or services for personal consumption",
}

# Ordinary inward categories (ITC eligible).
ELIGIBLE_CATEGORIES = {
    "raw_material": "Raw materials",
    "office_supplies": "Office supplies",
    "software_subscription": "Software subscription",
    "freight_courier": "Courier services",
    "machinery": "Plant and machinery",
}

# Inward services taxed under reverse charge (recipient pays tax in cash, then
# claims the same amount as ITC). Simplified subset of notification 13/2017.
RCM_CATEGORIES = {
    "legal_services": ("Legal services by an advocate", 18),
    "goods_transport_agency": ("Goods transport agency", 5),
    "sponsorship": ("Sponsorship services", 18),
}

# Taxable values are multiples of this, so every tax head is a whole rupee.
VALUE_STEP = 200
