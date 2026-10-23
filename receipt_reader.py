#!/usr/bin/env python3
"""
Extract items and prices from photos of grocery receipts
(Supermarché Match and Carrefour layouts).

Usage:
    python receipt_to_items.py photo1.jpg [photo2.jpg ...] [-o spending.csv]
                               [--date DD/MM/YY] [--debug]

For each photo it prints the items found, warns if they don't add up to the
receipt total, and appends one row per item to a CSV:
  price    = line amount as printed (for "3 X 2.19" lines: 6.57, with qty/unit_price)
  discount = immediate discount ("remise immédiate") applied to that product
  net      = price - discount, i.e. what the product actually cost you

Requirements:
    pip install rapidocr onnxruntime opencv-python
"""

import argparse
import csv
import datetime
import difflib
import os
import re
import sys
from decimal import Decimal

import cv2

from rapidocr import RapidOCR

# --------------------------------------------------------------------------- #
# Receipt layouts
# --------------------------------------------------------------------------- #

PRICE_RE = re.compile(r"^(-?\d{1,4})[.,](\d{2})\D{0,2}$")    # "44.36€", "1,95"
DATE_RE = re.compile(r"(\d{2})/(\d{2})/(\d{2})")
START_RE = re.compile(r"^BONJOUR|\bARTICLES\b")         # start of the item list
STORES = ["CARREFOUR", "MATCH", "LECLERC", "AUCHAN", "INTERMARCHE", "LIDL", "ALDI",
          "MONOPRIX", "CORA", "SUPER U", "HYPER U"]
TOTAL_RE = re.compile(r"^TOTAL\b\s*(.*)$")                   # "TOTAL", "TOTAL HORS AVANTAGES"
# quantity line: "3 X 2.19", "2 × 2.99 €", or weighed: "0.512 KG X 3.99"
QTY_RE = re.compile(r"^(\d+(?:[.,]\d{1,3})?)\s*(?:KG)?\s*[X×*]\s*(\d+)[.,](\d{2})")

CSV_HEADER = ["date", "store", "category", "item", "qty", "unit_price",
              "price", "discount", "net", "file"]

# --------------------------------------------------------------------------- #
# OCR -> text lines
# --------------------------------------------------------------------------- #

_engine = None


def run_ocr(img):
    """Return [(box_points, text), ...]."""
    global _engine
    if _engine is None:
        _engine = RapidOCR(params={"Global.log_level": "warning"})
    out = _engine(img)
    return list(zip(out.boxes, out.txts)) if out.txts else []


def ocr_lines(img):
    """Group OCR text boxes into printed lines (lists of strings, left to right)."""
    boxes = []
    for pts, text in run_ocr(img):
        ys = [p[1] for p in pts]
        boxes.append(((min(ys) + max(ys)) / 2, max(ys) - min(ys), pts[0][0], text.strip()))
    boxes.sort()
    lines, current, last_y = [], [], None
    for y, h, x, text in boxes:
        if last_y is not None and abs(y - last_y) > 0.5 * h:
            lines.append(current)
            current = []
        current.append((x, text))
        last_y = y
    if current:
        lines.append(current)
    return [[t for _, t in sorted(l)] for l in lines]


def price_of(text):
    m = PRICE_RE.match(text.replace(" ", ""))
    return Decimal(f"{m.group(1)}.{m.group(2)}") if m else None


def clean(text):
    text = re.sub(r"^[^A-Z0-9]+", "", text.upper())
    return re.sub(r"\s{2,}", " ", text).strip(" .,-~:|<>«»")


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def parse_lines(lines):
    """Read the ticket section by section:
      header -> items ("Articles ... Prix") -> totals & payments (up to TVA)
      -> immediate discounts ("Vous avez obtenu un avantage ...",
         "PROMOTIONS") -> rest
    Two quantity layouts are handled:
      Match:      "PATE TRES FEUIL 28" / "3 X 2.19   6.57"   (qty line carries the price)
      Carrefour:  "D 230G PAT FEUIL  4.17 €" / "3 x 1.39 €" (qty line after the price)
    """
    items, discounts, payments, totals, date = [], [], [], {}, None
    section, category, store = "header", "", ""
    pending = None      # line without a price: product name or department header

    for texts in lines:
        full = clean(" ".join(texts))
        price = price_of(texts[-1]) if len(texts) > 1 else None
        label = clean(" ".join(texts[:-1])) if price is not None else full

        m = DATE_RE.search(" ".join(texts))
        if m and not date:
            date = m.groups()

        if section == "header":
            store = store or next((s for s in STORES if re.search(rf"\b{s}\b", full)), "")
        if START_RE.search(full):                # "Bonjour ..." / "Code Articles Prix"
            if section == "header":
                section = "items"
            continue
        if section == "header":
            continue
        if re.fullmatch(r"\(\d+[.,]\d{2}\)", texts[-1].strip()):
            continue                             # "TRAITEUR TRAD (6.06)": info line only

        qm = QTY_RE.match(full)
        if qm:
            qty = Decimal(qm.group(1).replace(",", "."))
            unit = Decimal(f"{qm.group(2)}.{qm.group(3)}")
            # Carrefour style: the line shows only qty x unit price, the
            # product line above already has the total
            qty_after = price is None or (price == unit and qty != 1)
        if pending is not None and not (qm and not qty_after):
            # not followed by a quantity line -> it was a department header
            category, pending = pending, None

        tm = TOTAL_RE.match(label)
        if tm and section in ("items", "totals"):
            rest = tm.group(1)
            if price is not None:
                if "HORS" in rest:
                    totals["gross"] = price                  # before discounts
                elif "AVANTAGE" in rest or "REMISE" in rest:
                    totals["discount"] = abs(price)
                elif not rest:
                    totals["total"] = price                  # amount due
            section = "totals"
            continue

        if section == "totals":
            if "TVA" in full:
                section = "other"
            elif price is not None:
                payments.append(-price if label.startswith("RENDU") else price)
            continue

        if section in ("other", "discounts"):
            starts = ("REMISES" in full or label == "PROMOTIONS"
                      or ("AVANTAGE" in full and "OBTENU" in full))
            if starts:
                section = "discounts"            # the heading line is a subtotal: skip it
            elif section == "discounts":
                if tm or "CAGNOTTE" in full:
                    section = "other"            # end of discount list
                elif price is not None and label:
                    discounts.append((label, abs(price)))
            continue

        # ---- section == "items"
        if qm and qty_after:
            if items and items[-1].get("qty") is None:
                items[-1].update(qty=qty, unit=unit)
        elif qm:    # "3 X 2.19   6.57": the product name was on the line above
            items.append({"category": category, "item": pending or "?", "price": price,
                          "qty": qty, "unit": unit})
            pending = None
        elif price is None:
            pending = label or None
        elif label:
            if store == "CARREFOUR":
                label = re.sub(r"^[A-Z]\s+", "", label)   # VAT code letter "D ", "F "
            items.append({"category": category, "item": label, "price": price})

    return items, discounts, payments, totals, date, store.title()


def attach_discounts(items, discounts):
    """Put each immediate discount on the product it names.
    Returns the discounts that match no product."""
    names = [it["item"] for it in items]
    unmatched = []
    for name, amount in discounts:
        m = difflib.get_close_matches(name, names, n=1, cutoff=0.6)
        if m:
            it = items[names.index(m[0])]
            it["discount"] = it.get("discount", Decimal("0")) + amount
        else:
            unmatched.append((name, amount))
    return unmatched


def format_date(parts):
    if not parts:
        return ""
    d, mo, y = parts
    try:
        return datetime.date(2000 + int(y), int(mo), int(d)).isoformat()
    except ValueError:
        return ""


def check(items, unmatched, payments, totals):
    """Return a list of warnings about amounts that don't add up."""
    warnings = []
    s = sum((it["price"] for it in items if it["price"] is not None), Decimal("0"))
    disc = sum((it.get("discount", Decimal("0")) for it in items), Decimal("0")) \
        + sum((a for _, a in unmatched), Decimal("0"))
    total = totals.get("total")
    gross = totals.get("gross", total + totals.get("discount", 0) if total is not None else None)

    for it in items:
        if it["price"] is None:
            warnings.append(f"no price read for {it['item']}")
        elif it.get("qty") is not None and (it["qty"] * it["unit"]).quantize(Decimal("0.01")) != it["price"]:
            warnings.append(f"{it['item']}: {it['qty']:g} x {it['unit']} != {it['price']}")
    if gross is None:
        warnings.append("TOTAL not found on the photo")
    elif s != gross:
        warnings.append(f"items add up to {s}, receipt total is {gross} "
                        f"(difference {gross - s:+})")
    if "discount" in totals and disc != totals["discount"]:
        warnings.append(f"discounts found add up to {disc}, receipt says {totals['discount']}")
    if payments and total is not None and sum(payments) != total:
        warnings.append(f"payments add up to {sum(payments)}, TOTAL is {total}")
    return warnings


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_date_arg(s):
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            pass
    raise argparse.ArgumentTypeError("use DD/MM/YY, DD/MM/YYYY or YYYY-MM-DD")


def fr(d):
    """Decimal -> '1,95' for French spreadsheets ('' for None)."""
    return "" if d is None else str(d).replace(".", ",")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="receipt photos")
    ap.add_argument("-o", "--output", default="spending.csv",
                    help="CSV file to append to (default: spending.csv)")
    ap.add_argument("--date", type=parse_date_arg,
                    help="purchase date to use when it isn't readable on the photo")
    ap.add_argument("--debug", action="store_true", help="print the raw OCR lines")
    args = ap.parse_args()

    new_file = not os.path.exists(args.output)
    if not new_file:
        with open(args.output, encoding="utf-8") as f:
            if f.readline().strip().split(";") != CSV_HEADER:
                sys.exit(f"{args.output} has a different column layout; "
                         "rename it or choose another file with -o")

    with open(args.output, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")    # ';' opens cleanly in French Excel/LibreOffice
        if new_file:
            writer.writerow(CSV_HEADER)

        for path in args.images:
            name = os.path.basename(path)
            print(f"\n=== {name}")
            img = cv2.imread(path)
            if img is None:
                print("  ! cannot read this image")
                continue
            lines = ocr_lines(img)
            if args.debug:
                for l in lines:
                    print("   OCR|", "  |  ".join(l))

            items, discounts, payments, totals, date, store = parse_lines(lines)
            unmatched = attach_discounts(items, discounts)
            date = format_date(date) or args.date or ""

            for it in items:
                p = "   ????" if it["price"] is None else f"{it['price']:>7}"
                extra = f"  ({it['qty']:g} x {it['unit']})" if it.get("qty") is not None else ""
                if it.get("discount"):
                    extra += f"  discount -{it['discount']}"
                print(f"  {it['category'][:20]:<20} {it['item']:<24} {p}{extra}")
            for label, amount in unmatched:
                print(f"  {'':<20} {'REMISE ' + label:<24} {-amount:>7}")
            if totals.get("total") is not None:
                print(f"  {'':<20} {'TOTAL':<24} {totals['total']:>7}")
            print(f"  Store: {store or '?'}   Date: {date or '?'}")

            warnings = check(items, unmatched, payments, totals)
            if not date:
                warnings.append("date not found on the photo - rerun with --date DD/MM/YY")
            for w in warnings:
                print(f"  ! {w}")
            if not warnings:
                print("  OK: items add up to the receipt total")

            for it in items:
                disc = it.get("discount")
                net = None if it["price"] is None else it["price"] - (disc or 0)
                writer.writerow([date, store, it["category"], it["item"],
                                 fr(it.get("qty")), fr(it.get("unit")), fr(it["price"]),
                                 fr(disc), fr(net), name])
            for label, amount in unmatched:
                writer.writerow([date, store, "", "REMISE " + label, "", "", "",
                                 fr(amount), fr(-amount), name])
    print(f"\nRows appended to {args.output}")


if __name__ == "__main__":
    main()
