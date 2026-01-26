# pid2sid consumers (non-test)

| 路径 | 任务 | 输入格式要求 | 输出 schema |
| --- | --- | --- | --- |
| data/onerec_data/sft/video_rec.py | 视频推荐 SFT：根据历史观看预测下一个视频 | metadata parquet: `hist_video_pid`, `target_video_pid`, `uid`, `split` | `source`, `uuid`, `messages`, `metadata` |
| data/onerec_data/sft/label_cond_rec.py | 条件推荐 SFT：按行为类型预测交互内容 | metadata parquet: `hist_video_pid`, `target_video_pid`, `hist_video_{interaction}`, `target_video_{interaction}`, `uid`, `split` | `source`, `uuid`, `messages`, `metadata` |
| data/onerec_data/sft/product_rec.py | 跨域商品推荐 SFT：视频观看+商品点击预测商品 | metadata parquet: `hist_goods_pid`, `target_goods_pid`, `hist_longview_video_list`, `uid`, `split` | `source`, `uuid`, `messages`, `metadata` |
| data/onerec_data/sft/interactive_rec.py | 交互式推荐 SFT：用户画像+关键词预测交互内容 | metadata parquet: `inter_user_profile_with_sid`, `inter_keyword_to_items`, `uid`, `split` | `source`, `uuid`, `messages`, `metadata` |
| data/onerec_data/sft/ad_rec.py | 广告推荐 SFT：视频观看+广告点击预测广告 | metadata parquet: `hist_ad_pid`, `target_ad_pid`, `hist_longview_video_list`, `uid`, `split` | `source`, `uuid`, `messages`, `metadata` |
| data/onerec_data/sft/label_pred.py | 点式分类 SFT：预测候选视频是否“长观看” | metadata parquet: `hist_video_pid`, `target_video_pid`, `hist_video_{interaction}`, `target_video_longview`, `uid`, `split` | `source`, `uuid`, `messages`, `metadata` |
| data/onerec_data/sft/item_understand.py | 物品理解 SFT：给定 SID 生成描述 | caption parquet: `pid`, `dense_caption` | `source`, `uuid`, `messages`, `metadata` |
| data/onerec_data/pretrain/video_rec.py | 视频推荐预训练：拼接历史 SID 和目标 SID | metadata parquet: `hist_video_pid`, `target_video_pid`, `uid`, `split` | `source`, `uuid`, `segments`, `metadata` |
| data/onerec_data/pretrain/item_understand.py | 物品理解预训练：SID + caption 模板化 | caption parquet: `pid`, `dense_caption` | `source`, `uuid`, `segments`, `metadata` |
