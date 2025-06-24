import re
from typing import Dict, Any, List
from utils.database import MongoDB
from itertools import permutations

class DocumentFormatter:
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text by removing spaces and special characters"""
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    @staticmethod
    def compare_texts(text1: str, text2: str) -> bool:
        """Compare two texts ignoring spaces and special characters"""
        return DocumentFormatter.normalize_text(text1) == DocumentFormatter.normalize_text(text2)

    @staticmethod
    def generate_name_permutations(name_parts: List[str]) -> List[str]:
        """Generate all possible permutations of name parts"""
        # Remove empty strings and normalize
        name_parts = [part.strip() for part in name_parts if part.strip()]
        
        # Generate all possible permutations
        all_permutations = []
        for r in range(1, len(name_parts) + 1):
            perms = permutations(name_parts, r)
            all_permutations.extend([' '.join(perm) for perm in perms])
        
        return all_permutations

    @staticmethod
    def compare_names(extracted_name: str, db_name: Dict[str, str]) -> bool:
        """Compare extracted name with database name using permutations"""
        try:
            # Normalize extracted name
            extracted_name = extracted_name.strip().upper()
            
            # Get name parts from database
            first_name = db_name.get('first_name', '').strip().upper()
            last_name = db_name.get('last_name', '').strip().upper()
            middle_name = db_name.get('middle_name', '').strip().upper()
            
            # Create list of name parts
            name_parts = [part for part in [first_name, middle_name, last_name] if part]
            
            # Generate all possible permutations
            name_permutations = DocumentFormatter.generate_name_permutations(name_parts)
            
            # Also try the full name as is
            full_name = ' '.join(name_parts)
            name_permutations.append(full_name)
            
            # Compare extracted name with all permutations
            for perm in name_permutations:
                if DocumentFormatter.compare_texts(extracted_name, perm):
                    return True
                
                # Try comparing individual parts
                if any(DocumentFormatter.compare_texts(extracted_name, part) for part in name_parts):
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error in name comparison: {str(e)}")
            return False

    @staticmethod
    def format_aadhar_data(extracted_text: Dict[str, Any]) -> Dict[str, str]:
        """Format Aadhar card data into structured format and verify against database (flat fields only)"""
        data = {
            "aadhar_number": "",
            "name": "",
            "dob": "",
            "gender": "",
            "address": "",
            "is_verified": False,
            "verification_details": {}
        }
        
        try:
            # Extract Aadhar number
            if "aadhar no" in extracted_text:
                aadhar_text = extracted_text["aadhar no"]["text"]
                aadhar_match = re.search(r'\d{4}\s?\d{4}\s?\d{4}', aadhar_text)
                if aadhar_match:
                    data["aadhar_number"] = aadhar_match.group().replace(" ", "")
                    print("Extracted aadhar_number:", data["aadhar_number"])
                    
                    # Check MongoDB for existing document
                    mongo = MongoDB()
                    aadhar_collection = mongo.get_collection('aadhar')
                    
                    # Try exact match first
                    existing_doc = aadhar_collection.find_one({
                        "aadhar_number": data["aadhar_number"]
                    })
                    print("DB Query Result (exact):", existing_doc)
                    
                    # Fallback: Try regex match if not found (to handle minor OCR errors)
                    if not existing_doc:
                        regex_pattern = re.compile(r"^" + re.escape(data["aadhar_number"]) + r"$", re.IGNORECASE)
                        existing_doc = aadhar_collection.find_one({
                            "aadhar_number": {"$regex": regex_pattern}
                        })
                        print("DB Query Result (regex fallback):", existing_doc)
                    
                    if existing_doc:
                        data["is_verified"] = True
                        data["verification_details"] = {
                            "found_in_database": True,
                            "matches": {}
                        }
            
            # Extract other details
            if "details" in extracted_text:
                details_text = extracted_text["details"]["text"]
                
                # Split into lines and clean
                lines = [line.strip() for line in details_text.split('\n') if line.strip()]
                
                name_found = False
                address_lines = []
                for line in lines:
                    # First non-empty line is the name
                    if not name_found:
                        data["name"] = line
                        name_found = True
                        continue
                    # Check for DOB
                    dob_match = re.search(r'\d{2}/\d{2}/\d{4}', line)
                    if dob_match and not data["dob"]:
                        data["dob"] = dob_match.group()
                        continue
                    # Check for Gender
                    if not data["gender"]:
                        if any(gender in line.lower() for gender in ["female", "स्त्री", "महिला", "f"]):
                            data["gender"] = "Female"
                            continue
                        elif any(gender in line.lower() for gender in ["male", "पुरुष", "नर", "m"]):
                            data["gender"] = "Male"
                            continue
                    # If not DOB or gender, treat as address
                    address_lines.append(line)
                # Join address lines (if any)
                if address_lines:
                    data["address"] = ' '.join(address_lines).strip()
                    data["address"] = re.sub(r'\s+', ' ', data["address"])
                    data["address"] = re.sub(r'[^a-zA-Z0-9\s]', '', data["address"])
            
            # If we found an existing document, compare the details (flat fields)
            if data["is_verified"] and existing_doc:
                # Flexible text comparison for all fields
                def flex_compare(a, b):
                    return DocumentFormatter.normalize_text(a) == DocumentFormatter.normalize_text(b)
                
                matches = {}
                matches["name"] = flex_compare(data["name"], existing_doc.get("name", ""))
                matches["dob"] = flex_compare(data["dob"], existing_doc.get("dob", ""))
                matches["gender"] = flex_compare(data["gender"], existing_doc.get("gender", ""))
                matches["address"] = flex_compare(data["address"], existing_doc.get("address", ""))
                data["verification_details"]["matches"] = matches
                
                # Calculate overall match percentage
                match_count = sum(1 for match in matches.values() if match)
                total_fields = len(matches)
                data["verification_details"]["match_percentage"] = (match_count / total_fields) * 100 if total_fields > 0 else 0
                
                # Add raw details for debugging
                data["verification_details"]["raw_details"] = {
                    "extracted": {
                        "name": data["name"],
                        "dob": data["dob"],
                        "gender": data["gender"],
                        "address": data["address"]
                    },
                    "database": {
                        "name": existing_doc.get("name", ""),
                        "dob": existing_doc.get("dob", ""),
                        "gender": existing_doc.get("gender", ""),
                        "address": existing_doc.get("address", "")
                    }
                }
            
            return data
            
        except Exception as e:
            print(f"Error in Aadhar data formatting: {str(e)}")
            data["is_verified"] = False
            data["verification_details"] = {
                "error": str(e)
            }
            return data

    @staticmethod
    def format_pan_data(extracted_text: Dict[str, Any]) -> Dict[str, str]:
        """Format PAN card data into structured format and verify against database (flat fields only)"""
        data = {
            "pan_number": "",
            "name": "",
            "father_name": "",
            "dob": "",
            "is_verified": False,
            "verification_details": {}
        }
        try:
            # Extract PAN number
            if "pan" in extracted_text:
                pan_text = extracted_text["pan"]["text"]
                pan_match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', pan_text)
                if pan_match:
                    data["pan_number"] = pan_match.group()
                    print("Extracted pan_number:", data["pan_number"])
                    # Check MongoDB for existing document
                    mongo = MongoDB()
                    pan_collection = mongo.get_collection('pan')
                    # Try exact match first
                    existing_doc = pan_collection.find_one({
                        "pan_number": data["pan_number"]
                    })
                    print("DB Query Result (exact):", existing_doc)
                    # Fallback: Try regex match if not found
                    if not existing_doc:
                        regex_pattern = re.compile(r"^" + re.escape(data["pan_number"]) + r"$", re.IGNORECASE)
                        existing_doc = pan_collection.find_one({
                            "pan_number": {"$regex": regex_pattern}
                        })
                        print("DB Query Result (regex fallback):", existing_doc)
                    if existing_doc:
                        data["is_verified"] = True
                        data["verification_details"] = {
                            "found_in_database": True,
                            "matches": {}
                        }
            # Extract other details
            if "details" in extracted_text:
                details_text = extracted_text["details"]["text"]
                dob_match = re.search(r'\d{2}/\d{2}/\d{4}', details_text)
                if dob_match:
                    data["dob"] = dob_match.group()
                lines = details_text.split('\n')
                for line in lines:
                    if not any(keyword in line.lower() for keyword in ["dob", "date of birth"]):
                        if not data["name"]:
                            data["name"] = line.strip()
                        elif not data["father_name"]:
                            data["father_name"] = line.strip()
            # If we found an existing document, compare the details (flat fields)
            if data["is_verified"] and existing_doc:
                def flex_compare(a, b):
                    return DocumentFormatter.normalize_text(a) == DocumentFormatter.normalize_text(b)
                def flex_date_compare(extracted, db):
                    # Try to match dd/mm/yyyy (OCR) with yyyy-mm-dd (DB)
                    import re
                    def normalize_date(date):
                        if re.match(r'\d{2}/\d{2}/\d{4}', date):
                            d, m, y = date.split('/')
                            return f"{y}-{m}-{d}"
                        return date
                    return normalize_date(extracted) == db
                matches = {}
                matches["name"] = flex_compare(data["name"], existing_doc.get("name", ""))
                matches["father_name"] = flex_compare(data["father_name"], existing_doc.get("father_name", ""))
                matches["dob"] = flex_date_compare(data["dob"], existing_doc.get("date_of_birth", ""))
                data["verification_details"]["matches"] = matches
                match_count = sum(1 for match in matches.values() if match)
                total_fields = len(matches)
                data["verification_details"]["match_percentage"] = (match_count / total_fields) * 100 if total_fields > 0 else 0
                data["verification_details"]["raw_details"] = {
                    "extracted": {
                        "name": data["name"],
                        "father_name": data["father_name"],
                        "dob": data["dob"]
                    },
                    "database": {
                        "name": existing_doc.get("name", ""),
                        "father_name": existing_doc.get("father_name", ""),
                        "date_of_birth": existing_doc.get("date_of_birth", "")
                    }
                }
            return data
        except Exception as e:
            print(f"Error in PAN data formatting: {str(e)}")
            data["is_verified"] = False
            data["verification_details"] = {
                "error": str(e)
            }
            return data

    @staticmethod
    def format_driving_license_data(extracted_text: Dict[str, Any]) -> Dict[str, str]:
        """Format Driving License data into structured format and verify against database (flat fields only)"""
        data = {
            "license_number": "",
            "name": "",
            "dob": "",
            "address": "",
            "issue_date": "",
            "expiry_date": "",
            "class": "",
            "sex": "",
            "is_verified": False,
            "verification_details": {}
        }
        try:
            # Extract License Number
            if "License number" in extracted_text:
                license_text = extracted_text["License number"]["text"]
                license_text = re.sub(r'[^A-Z0-9]', '', license_text)
                data["license_number"] = license_text
                print("Extracted license_number:", data["license_number"])
                # Check MongoDB for existing document
                mongo = MongoDB()
                dl_collection = mongo.get_collection('driving_license')
                existing_doc = dl_collection.find_one({
                    "license_number": data["license_number"]
                })
                print("DB Query Result (exact):", existing_doc)
                if not existing_doc:
                    regex_pattern = re.compile(r"^" + re.escape(data["license_number"]) + r"$", re.IGNORECASE)
                    existing_doc = dl_collection.find_one({
                        "license_number": {"$regex": regex_pattern}
                    })
                    print("DB Query Result (regex fallback):", existing_doc)
                if existing_doc:
                    data["is_verified"] = True
                    data["verification_details"] = {
                        "found_in_database": True,
                        "matches": {}
                    }
            # Extract other details (same as before)
            if "First name" in extracted_text and "Last name" in extracted_text:
                first_name = extracted_text["First name"]["text"].strip()
                last_name = extracted_text["Last name"]["text"].strip()
                first_name = ' '.join(dict.fromkeys(first_name.split()))
                last_name = ' '.join(dict.fromkeys(last_name.split()))
                data["name"] = f"{first_name} {last_name}".strip()
            if "DOB" in extracted_text:
                dob_text = extracted_text["DOB"]["text"]
                dob_match = re.search(r'\d{2}/\d{2}/\d{4}', dob_text)
                if dob_match:
                    data["dob"] = dob_match.group()
            if "Address" in extracted_text:
                address_text = extracted_text["Address"]["text"]
                address_text = re.sub(r'\s+', ' ', address_text)
                address_parts = []
                seen = set()
                for part in address_text.split():
                    if part not in seen:
                        seen.add(part)
                        address_parts.append(part)
                data["address"] = ' '.join(address_parts)
            if "Issue date" in extracted_text:
                issue_text = extracted_text["Issue date"]["text"]
                issue_match = re.search(r'\d{2}/\d{2}/\d{4}', issue_text)
                if issue_match:
                    data["issue_date"] = issue_match.group()
            if "Exp date" in extracted_text:
                exp_text = extracted_text["Exp date"]["text"]
                exp_match = re.search(r'\d{2}/\d{2}/\d{4}', exp_text)
                if exp_match:
                    data["expiry_date"] = exp_match.group()
            if "Class" in extracted_text:
                class_text = extracted_text["Class"]["text"]
                class_text = re.sub(r'[^A-Z0-9]', '', class_text)
                data["class"] = class_text
            if "Sex" in extracted_text:
                sex_text = extracted_text["Sex"]["text"]
                sex_text = sex_text.upper()
                if any(gender in sex_text for gender in ["F", "FEMALE"]):
                    data["sex"] = "F"
                elif any(gender in sex_text for gender in ["M", "MALE"]):
                    data["sex"] = "M"
            # If we found an existing document, compare the details (flat fields)
            if data["is_verified"] and existing_doc:
                def flex_compare(a, b):
                    return DocumentFormatter.normalize_text(a) == DocumentFormatter.normalize_text(b)
                matches = {}
                matches["name"] = flex_compare(data["name"], existing_doc.get("name", ""))
                matches["dob"] = flex_compare(data["dob"], existing_doc.get("dob", ""))
                matches["address"] = flex_compare(data["address"], existing_doc.get("address", ""))
                matches["issue_date"] = flex_compare(data["issue_date"], existing_doc.get("issue_date", ""))
                matches["expiry_date"] = flex_compare(data["expiry_date"], existing_doc.get("expiry_date", ""))
                matches["class"] = flex_compare(data["class"], existing_doc.get("class", ""))
                matches["sex"] = flex_compare(data["sex"], existing_doc.get("sex", ""))
                data["verification_details"]["matches"] = matches
                match_count = sum(1 for match in matches.values() if match)
                total_fields = len(matches)
                data["verification_details"]["match_percentage"] = (match_count / total_fields) * 100 if total_fields > 0 else 0
                data["verification_details"]["raw_details"] = {
                    "extracted": {
                        "name": data["name"],
                        "dob": data["dob"],
                        "address": data["address"],
                        "issue_date": data["issue_date"],
                        "expiry_date": data["expiry_date"],
                        "class": data["class"],
                        "sex": data["sex"]
                    },
                    "database": {
                        "name": existing_doc.get("name", ""),
                        "dob": existing_doc.get("dob", ""),
                        "address": existing_doc.get("address", ""),
                        "issue_date": existing_doc.get("issue_date", ""),
                        "expiry_date": existing_doc.get("expiry_date", ""),
                        "class": existing_doc.get("class", ""),
                        "sex": existing_doc.get("sex", "")
                    }
                }
            return data
        except Exception as e:
            print(f"Error in Driving License data formatting: {str(e)}")
            data["is_verified"] = False
            data["verification_details"] = {
                "error": str(e)
            }
            return data

    @classmethod
    def format_document_data(cls, card_type: str, extracted_text: Dict[str, Any]) -> Dict[str, str]:
        """Format extracted text based on document type"""
        if card_type == "Aadhar":
            return cls.format_aadhar_data(extracted_text)
        elif card_type == "PAN":
            return cls.format_pan_data(extracted_text)
        elif card_type == "Driving License":
            return cls.format_driving_license_data(extracted_text)
        else:
            return extracted_text 