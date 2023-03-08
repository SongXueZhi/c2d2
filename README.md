# delta-debugging

Build your images by command
```docker build --tag ddchanges .```


select t.bic, t.id
from  dd_result t1 
inner join dd_result t2 
inner join regressions_all t
on t1.tool = "prodd" and t2.tool = "ddmin" and t1.regression_id = t2.regression_id and t2.regression_id = t.id and t1.cc_size - t2.cc_size < 0
