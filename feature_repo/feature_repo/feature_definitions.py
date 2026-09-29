from datetime import timedelta
from feast import (
    Entity,
    Field,
    FeatureView,
    FileSource,
    ValueType,
)
from feast.types import Float64

credit_card_source = FileSource(
    name="credit_card_source",
    path="../../data/processed/full_data.parquet",  # We will generate this unified file next
    timestamp_field="event_timestamp",
)

transaction = Entity(
    name="transaction_id", 
    value_type=ValueType.INT64,
    join_keys=["transaction_id"]
)

# Build dynamic schema for V1-V28 and Amount
schema_fields = [Field(name=f"V{i}", dtype=Float64) for i in range(1, 29)]
schema_fields.append(Field(name="Amount", dtype=Float64))

credit_card_fv = FeatureView(
    name="credit_card_features",
    entities=[transaction],
    ttl=None,
    schema=schema_fields,
    online=True,
    source=credit_card_source,
)