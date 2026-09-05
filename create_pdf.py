import fitz
doc = fitz.open()
page = doc.new_page()
page.insert_text((50, 50), "This is a dummy PDF file for testing TrustExtract-N.")
page.insert_text((50, 70), "The Ministry of Finance issued a notice today.")
page.insert_text((50, 90), "Date: 12 Jan 2024. Contact: test@example.com.")
doc.save("trustextract-n/data/dummy.pdf")
doc.close()
