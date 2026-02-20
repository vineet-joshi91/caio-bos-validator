# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 17:52:36 2026

@author: Vineet
"""

#!/usr/bin/env python3
"""
Add credit packs to database for Razorpay payment pages.
Run once to populate the credit_packs table.
"""

from db import get_db
from wallet import CreditPack

def add_credit_packs():
    """Add the 3 credit packs: Starter, Growth, Pro"""
    
    db = next(get_db())
    
    packs = [
        {
            "pack_id": "starter_120",
            "currency": "INR",
            "amount_minor_units": 99900,  # ₹999.00 in paise
            "credits": 120,
            "display_name": "Starter Pack",
            "description": "120 credits for CAIO analysis (2 EA+DR combos or 12 EA-only runs)",
            "is_active": True,
            "gateway_product_id": "caio-starter-120",  # Matches Razorpay page slug
        },
        {
            "pack_id": "growth_300",
            "currency": "INR",
            "amount_minor_units": 199900,  # ₹1,999.00 in paise
            "credits": 300,
            "display_name": "Growth Pack",
            "description": "300 credits for CAIO analysis (5 EA+DR combos or 30 EA-only runs)",
            "is_active": True,
            "gateway_product_id": "caio-growth-300",
        },
        {
            "pack_id": "pro_600",
            "currency": "INR",
            "amount_minor_units": 399900,  # ₹3,999.00 in paise
            "credits": 600,
            "display_name": "Pro Pack",
            "description": "600 credits for CAIO analysis (10 EA+DR combos or 60 EA-only runs)",
            "is_active": True,
            "gateway_product_id": "caio-pro-600",
        },
    ]
    
    for pack_data in packs:
        # Check if pack already exists
        existing = db.query(CreditPack).filter(
            CreditPack.pack_id == pack_data["pack_id"]
        ).first()
        
        if existing:
            print(f"✓ Pack '{pack_data['pack_id']}' already exists, skipping...")
            continue
        
        # Create new pack
        pack = CreditPack(**pack_data)
        db.add(pack)
        print(f"✓ Added pack: {pack_data['display_name']} ({pack_data['credits']} credits)")
    
    db.commit()
    print("\n✅ All credit packs added successfully!")
    print("\nPacks in database:")
    
    all_packs = db.query(CreditPack).filter(CreditPack.is_active == True).all()
    for p in all_packs:
        print(f"  - {p.display_name}: {p.credits} credits for ₹{p.amount_minor_units / 100:.2f}")

if __name__ == "__main__":
    add_credit_packs()